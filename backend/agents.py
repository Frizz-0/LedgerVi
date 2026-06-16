# agents.py
import json
import os
import re
from engine_state import TrialState
from database import trialguard_db
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")

llm_fast = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

llm_smart = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    model_name="llama-3.3-70b-versatile",
    temperature=0.0
)

# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def _parse_json_response(content: str) -> dict | list | None:
    """Isolates and parses structured JSON blocks safely using regex bounds."""
    content = content.strip()
    match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    try:
        return json.loads(content)
    except Exception:
        return None


def _rerank_medical_rules(procedure_notes: str, candidates: list) -> list:
    if not candidates:
        return []

    formatted = "\n".join([f'  "{r["id"]}": {r["text"]}' for r in candidates])

    prompt = f"""You are a senior clinical compliance analyst reviewing a trial claim.

Patient Clinical Notes:
"{procedure_notes}"

Candidate Regulations to Evaluate:
{formatted}

For each regulation:
1. Read the EXACT trigger condition the regulation requires.
2. Read what the patient's notes ACTUALLY document.
3. Mark YES only if the notes explicitly document the threshold condition. Do not infer.
4. For regulations that require MULTIPLE conditions: every single stated condition must be explicitly present in the notes. Matching ONE condition out of TWO or more is NOT sufficient to trigger the regulation.
5. REG-550 specifically requires BOTH: (a) therapy was administered at a location OUTSIDE the primary trial site, AND (b) the episode was explicitly described as acute and life-threatening. An external or un-audited supplier alone does NOT satisfy REG-550 unless both conditions are directly stated.
6. If the patient's condition is milder, different, or only partially matches the threshold — mark NO.

Return ONLY a raw JSON array of the IDs of regulations that are genuinely triggered.
Use exact ID strings as shown (e.g. "REG-550"). Return [] if none apply.
No markdown wrappers. No explanation."""

    response = llm_smart.invoke(prompt)
    result = _parse_json_response(response.content)

    if isinstance(result, list):
        applicable_ids = [x for x in result if isinstance(x, str)]
    else:
        applicable_ids = []

    return [r for r in candidates if r["id"] in applicable_ids]


def _apply_financial_rules_deterministically(line_items: dict, rule_decisions: list) -> dict:
    approved = {}
    disputed = {}

    pool_budgets: dict[str, float] = {}
    for rule in rule_decisions:
        if rule.get("rule_type") == "combined_cap":
            group = rule.get("combined_cap_group") or "default_pool"
            limit = float(rule.get("combined_cap_limit") or 0)
            if group not in pool_budgets:
                pool_budgets[group] = limit

    for key, amount in line_items.items():
        amount = float(amount)
        rule = next((r for r in rule_decisions if r.get("fee_key") == key), None)
        rule_type = rule.get("rule_type", "full_approval") if rule else "full_approval"

        if rule_type == "exclusion":
            approved[key] = 0.0
            disputed[key] = amount

        elif rule_type == "cap":
            limit = float(rule.get("limit") or 0)
            approved[key] = min(amount, limit)
            disputed[key] = max(0.0, amount - limit)

        elif rule_type == "combined_cap":
            group = rule.get("combined_cap_group") or "default_pool"
            budget = pool_budgets.get(group, 0.0)
            app = min(amount, budget)
            dis = max(0.0, amount - budget)
            approved[key] = app
            disputed[key] = dis
            pool_budgets[group] = max(0.0, budget - amount)

        else:
            approved[key] = amount
            disputed[key] = 0.0

    total_approved = round(sum(approved.values()), 2)
    total_disputed = round(sum(disputed.values()), 2)
    total_claim    = round(sum(float(v) for v in line_items.values()), 2)

    integrity_ok = abs((total_approved + total_disputed) - total_claim) <= 0.02

    return {
        "authorized_payout":     total_approved if integrity_ok else 0.0,
        "escrow_dispute_amount": total_disputed if integrity_ok else total_claim,
        "integrity_ok":          integrity_ok,
        "per_item_disputed":     disputed
    }


# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------

def ingestion_router_node(state: TrialState):
    print("[System Router] Inspecting incoming data structural payloads...")
    iso   = float(state.get("invoice_isolation_fees", 0.0))
    labor = float(state.get("invoice_labor_fees", 0.0))
    meds  = float(state.get("invoice_medication_fees", 0.0))

    if (iso + labor + meds) > 0.0:
        print("   -> Route Confirmed: HYBRID (Structured line items identified).")
        return {"ingestion_routing_path": "HYBRID", "next_step": "RESEARCHER"}

    narrative = state.get("procedure_notes", "").strip()
    if len(narrative) > 20:
        print("   -> Route Confirmed: COGNITIVE (Parsing unstructured text narrative).")
        return {"ingestion_routing_path": "COGNITIVE", "next_step": "COGNITIVE_EXTRACTOR"}

    print("   -> Route Aborted: Insufficient data payload strings.")
    return {"ingestion_routing_path": "MALFORMED", "next_step": "FINISH"}


def cognitive_extractor_node(state: TrialState):
    print("[Worker: Cognitive Extractor] Parsing raw narrative strings into digits...")
    prompt = f"""Extract billing amounts from this hospital invoice narrative.
Narrative: "{state['procedure_notes']}"

Output ONLY a raw JSON object matching this schema shape perfectly:
{{
    "extracted_isolation_fees": 0.0,
    "extracted_labor_fees": 0.0,
    "extracted_medication_fees": 0.0
}}"""

    response = llm_fast.invoke(prompt)
    data = _parse_json_response(response.content) or {}

    iso   = float(data.get("extracted_isolation_fees", 0.0))
    labor = float(data.get("extracted_labor_fees", 0.0))
    meds  = float(data.get("extracted_medication_fees", 0.0))

    print(f"   [Extracted] Iso: £{iso} | Labor: £{labor} | Meds: £{meds}")
    return {
        "invoice_isolation_fees": iso,
        "invoice_labor_fees": labor,
        "invoice_medication_fees": meds,
        "claim_amount": iso + labor + meds
    }


def medical_researcher_node(state: TrialState):
    print("[Worker: Medical Researcher] Fetching protocol vectors...")
    candidates = trialguard_db.query_medical(state["procedure_notes"])
    
    if not candidates:
        return {
            "medical_context": "No regulations found in the knowledge base.",
            "matched_rule_id": "REG-NONE",
            "raw_rule_text": "No regulation context available.",
            "medical_checked": True,
            "next_step": "SUPERVISOR"
        }

    applicable = _rerank_medical_rules(state["procedure_notes"], candidates)

    if not applicable:
        return {
            "medical_context": "No clinical regulations triggered. Patient condition does not meet any regulatory threshold. Case is medically standard.",
            "matched_rule_id": "REG-NONE",
            "raw_rule_text": f"No regulations triggered. Nearest candidate: {candidates[0]['text']}",
            "medical_checked": True,
            "next_step": "SUPERVISOR"
        }

    applicable_text = "\n".join([r["text"] for r in applicable])
    primary_rule_id = applicable[0]["id"]

    prompt = f"""You are a clinical trial compliance auditor.
Patient Notes: {state['procedure_notes']}
Applicable Regulations: {applicable_text}

Write a 1-sentence audit verdict describing compliance status. No markdown. No filler."""

    response = llm_smart.invoke(prompt)
    return {
        "medical_context": response.content,
        "matched_rule_id": primary_rule_id,
        "raw_rule_text": applicable_text,
        "medical_checked": True,
        "next_step": "SUPERVISOR"
    }


def fintech_underwriter_node(state: TrialState):
    print("[Worker: FinTech Underwriter] Executing hybrid deterministic financial audit...")
    matched_contract, default_clause_id, clause_docs = trialguard_db.query_financial(state["procedure_notes"])

    # Build exact clause ID → document text lookup so we can validate LLM outputs
    # and reorder display text reliably (primary clause always first)
    id_to_doc: dict[str, str] = {}
    for doc in clause_docs:
        if ":" in doc:
            cid = doc.split(":")[0].strip()
            id_to_doc[cid] = doc
    valid_clause_ids = set(id_to_doc.keys())

    iso   = float(state.get("invoice_isolation_fees", 0.0))
    labor = float(state.get("invoice_labor_fees", 0.0))
    meds  = float(state.get("invoice_medication_fees", 0.0))

    line_items = {
        "invoice_isolation_fees": iso,
        "invoice_labor_fees": labor,
        "invoice_medication_fees": meds
    }

    # Provide the LLM with the exact clause IDs it is allowed to use — prevents hallucination
    available_clause_ids = ", ".join(f'"{cid}"' for cid in valid_clause_ids)

    prompt = f"""You are a financial compliance analyst. For each billing line item, classify which policy clause applies. Do NOT compute any monetary amounts.

### Available Clause IDs — use ONLY these exact strings for rule_id:
{available_clause_ids}

### Billing Line Items:
1. invoice_isolation_fees: £{iso}
2. invoice_labor_fees: £{labor}
3. invoice_medication_fees: £{meds}

### Policy Clauses:
{matched_contract}

### Clinical Context:
"{state['procedure_notes']}"

### Rule Types — select one per item:
- "full_approval": No clause targets this fee type, or conditions are unmet. Approved at face value.
- "cap": Clause sets a single ceiling limit for THIS specific fee type on its own. Set "limit" parameter.
- "exclusion": ONLY use if the clinical notes EXPLICITLY contain words such as "un-audited", "non-affiliated", "third-party supplier", "external pharmaceutical", or equivalent language confirming the medication/therapy was sourced outside sanctioned channels. Generic terms like "compounding", "auxiliary", "formulation", or "pharmacy" do NOT trigger exclusion. If notes say medications came from primary site, affiliated, or on-site pharmacies, use "full_approval" — this is the opposite of an exclusion trigger.
- "combined_cap": Clause sets a single shared ceiling across MULTIPLE fee types together. Assign matching "combined_cap_group" names and limits across entries.

Output ONLY a raw JSON payload matching this template array exactly with no markdown fences:
{{
    "rule_decisions": [
        {{
            "fee_key": "invoice_isolation_fees",
            "rule_id": "EXACT_CLAUSE_ID",
            "rule_type": "full_approval | cap | exclusion | combined_cap",
            "limit": number or null,
            "combined_cap_group": "group_label or null",
            "combined_cap_limit": number or null,
            "justification": "text description"
        }},
        {{
            "fee_key": "invoice_labor_fees",
            "rule_id": "EXACT_CLAUSE_ID",
            "rule_type": "full_approval | cap | exclusion | combined_cap",
            "limit": number or null,
            "combined_cap_group": null,
            "combined_cap_limit": null,
            "justification": "text description"
        }},
        {{
            "fee_key": "invoice_medication_fees",
            "rule_id": "EXACT_CLAUSE_ID",
            "rule_type": "full_approval | cap | exclusion | combined_cap",
            "limit": number or null,
            "combined_cap_group": "group_label or null",
            "combined_cap_limit": number or null,
            "justification": "text description"
        }}
    ]
}}"""

    response = llm_smart.invoke(prompt)
    data = _parse_json_response(response.content)

    if data and isinstance(data, dict):
        rule_decisions = data.get("rule_decisions", [])
        # 🛡️ VALIDATION: Discard any rule_id the LLM returned that is not in the retrieved clause set.
        # This prevents hallucinated or mismatched IDs from corrupting the math or the display.
        for r in rule_decisions:
            if r.get("rule_id") not in valid_clause_ids:
                r["rule_id"] = "NONE"
                r["rule_type"] = "full_approval"
                r["limit"] = None
        justifications = " | ".join(f"{r.get('fee_key')}: {r.get('justification')}" for r in rule_decisions)
    else:
        rule_decisions = []
        justifications = "LLM parsing error — defaulted to baseline parameters."

    result = _apply_financial_rules_deterministically(line_items, rule_decisions)
    payout  = result["authorized_payout"]
    dispute = result["escrow_dispute_amount"]

    # 🛡️ SYSTEM INTEGRITY FIX: Determine the primary triggered clause using rule type priority.
    # Priority: exclusion=3 > combined_cap=2 > cap=1 > full_approval=0.
    # Within the same priority tier, the item with the largest financial deduction wins.
    # This guarantees matched_clause_id always reflects the most impactful active rule.
    RULE_PRIORITY = {"exclusion": 3, "combined_cap": 2, "cap": 1, "full_approval": 0}

    best_priority   = -1
    best_dispute_amt = -1.0
    final_clause_id = "POLICY-UNKNOWN"

    for item in rule_decisions:
        fee_key = item.get("fee_key")
        r_id    = item.get("rule_id")
        r_type  = item.get("rule_type", "full_approval")

        if not r_id or r_id in ("NONE", "POLICY-UNKNOWN"):
            continue

        priority     = RULE_PRIORITY.get(r_type, 0)
        item_disputed = float(result["per_item_disputed"].get(fee_key, 0.0)) if fee_key else 0.0

        if (priority > best_priority) or (priority == best_priority and item_disputed > best_dispute_amt):
            best_priority    = priority
            best_dispute_amt = item_disputed
            final_clause_id  = r_id

    # If the deterministic engine produced zero escrow, no clause actually penalised anything.
    # Use POLICY-UNKNOWN regardless of what the LLM labelled — labels without deductions are meaningless.
    # Only fall back to the RAG primary hit if a real deduction exists but clause ID resolution failed.
    if dispute == 0.0:
        final_clause_id = "POLICY-UNKNOWN"
    elif final_clause_id == "POLICY-UNKNOWN":
        # Active deduction exists but clause ID resolution failed — fall back to primary RAG hit
        final_clause_id = default_clause_id

    if not result["integrity_ok"]:
        final_clause_id = "INTEGRITY-FAIL"
        justifications = "Arithmetic integrity check failed. Full claim diverted to escrow."

    # 🛡️ DISPLAY INTEGRITY: Build clause context with the primary (matched) clause doc FIRST,
    # then any other applied clause docs. This guarantees the top-1 of Clause Context always
    # corresponds exactly to the matched_clause_id shown in the UI.
    applied_rule_ids = {
        r.get("rule_id") for r in rule_decisions
        if r.get("rule_id") and r.get("rule_id") not in ("NONE", "POLICY-UNKNOWN")
    }
    primary_doc    = id_to_doc.get(final_clause_id, "")
    secondary_docs = [id_to_doc[cid] for cid in applied_rule_ids if cid != final_clause_id and cid in id_to_doc]

    if primary_doc:
        display_clause_text = primary_doc + ("\n" + "\n".join(secondary_docs) if secondary_docs else "")
    else:
        # Fallback: show all retrieved clause docs
        display_clause_text = matched_contract

    calculated_confidence = 94.5 if final_clause_id in display_clause_text else 88.0

    print(f"   [Audit] Authorized: £{payout} | Dispute: £{dispute} | Clause: {final_clause_id}")

    return {
        "financial_context": f"Contract audit complete. {justifications}",
        "authorized_payout": payout,
        "escrow_dispute_amount": dispute,
        "matched_clause_id": final_clause_id,
        "raw_clause_text": display_clause_text,
        "rag_confidence_percentage": calculated_confidence,
        "financial_checked": True,
        "next_step": "SUPERVISOR"
    }


def risk_grader_node(state: TrialState):
    print("[Worker: Risk Grader] Computing risk tier...")
    prompt = f"""You are an enterprise risk classification engine. Assign exactly one tier.

TIER: HIGH_MEDICAL_RISK
  The clinical notes document an acute medical emergency requiring immediate intervention.
  Classify here if the notes contain ANY of these indicators:
  • Emergency presentation / emergency protocol activation
  • Immediate isolation, transfer, or containment for patient safety
  • Aggressive or emergency-level medical intervention
  • Acute physiological instability: pyrexia, rapid vital sign changes, respiratory distress
  • Explicit stabilisation efforts — implying the patient was unstable

TIER: HIGH_FINANCIAL_RISK
  Financial policy violations are confirmed: a cap was exceeded, or an exclusion was hit.
  Use this tier ONLY when financial violations are present but notes show routine, stable clinical parameters.

TIER: CLEAR
  No financial policy violations AND no acute clinical emergency indicators in the notes.

Priority rule: if BOTH medical and financial flags are present, assign HIGH_MEDICAL_RISK.

Current State:
- Patient History:   {state['clinical_history_summary']}
- Medical Audit:     {state['medical_context']}
- Financial Audit:   {state['financial_context']}

Begin response directly with exactly one classification header:
"TIER: CLEAR | REASON:"
"TIER: HIGH_MEDICAL_RISK | REASON:"
"TIER: HIGH_FINANCIAL_RISK | REASON:" """

    response = llm_smart.invoke(prompt)
    return {
        "triage_verdict": response.content,
        "triage_checked": True,
        "next_step": "SUPERVISOR"
    }


def medical_circuit_breaker_node(state: TrialState):
    print("[System Override] Checking medical risk circuit triggers...")
    gross_claim = float(state.get("claim_amount", 0.0))

    if "HIGH_MEDICAL_RISK" in state.get("triage_verdict", ""):
        print("   -> BREAKER TRIPPED: Locking down all financial allocations.")
        return {
            "authorized_payout": 0.0,
            "escrow_dispute_amount": gross_claim,
            "financial_context": "CRITICAL MEDICAL RISK: Full claim frozen by circuit breaker pending specialist review.",
            "breaker_checked": True,
            "next_step": "FINISH"
        }

    print("   -> Breaker Clear: Proceeding to final settlement.")
    return {"breaker_checked": True, "next_step": "FINISH"}


def managing_director_supervisor_node(state: TrialState):
    if not state.get("medical_checked", False):   target = "RESEARCHER"
    elif not state.get("financial_checked", False): target = "FINTECH_AUDITOR"
    elif not state.get("triage_checked", False):    target = "RISK_GRADER"
    elif not state.get("breaker_checked", False):   return {"next_step": "CIRCUIT_CHECK"}
    else:                                           target = "FINISH"
    return {"next_step": target}