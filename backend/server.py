# server.py
import uvicorn
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

from engine_state import TrialState, ClaimIngestRequest, HumanSignoffRequest, MetricsResponse, EnterpriseResponse
from agents import (
    ingestion_router_node,
    cognitive_extractor_node,
    managing_director_supervisor_node,
    medical_researcher_node,
    fintech_underwriter_node,
    medical_circuit_breaker_node,
    risk_grader_node
)

load_dotenv()
DB_PARAMS = os.getenv("Postgres_URL")


if DB_PARAMS:
    print("API Key successfully loaded!")
else:
    print("Failed to load API Key. Check your .env file path.")

def init_relational_database():
    conn = None
    try:
        conn = psycopg2.connect(DB_PARAMS)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corporate_ledger (
                id SERIAL PRIMARY KEY,
                thread_id VARCHAR(100) UNIQUE,
                trial_id VARCHAR(100),
                patient_id VARCHAR(100),
                patient_name VARCHAR(150),
                patient_sex VARCHAR(50),
                patient_age INT,
                patient_cohort VARCHAR(150),
                patient_enrollment VARCHAR(50),
                patient_incident VARCHAR(50),
                clinical_history TEXT,
                invoice_isolation NUMERIC,
                invoice_labor NUMERIC,
                invoice_medication NUMERIC,
                claim_amount NUMERIC,
                payout_amount NUMERIC,
                dispute_amount NUMERIC,
                matched_rule VARCHAR(50),
                matched_clause VARCHAR(50),
                confidence_score NUMERIC,
                status VARCHAR(50),
                timestamp_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            ALTER TABLE corporate_ledger ADD COLUMN IF NOT EXISTS raw_clause_text TEXT;
            ALTER TABLE corporate_ledger ADD COLUMN IF NOT EXISTS system_verdict TEXT;
            ALTER TABLE corporate_ledger ADD COLUMN IF NOT EXISTS raw_rule_text TEXT;
        """)
        conn.commit()
        print("PostgreSQL storage fully synced.")
    except Exception as e:
        print(f"PostgreSQL engine unreachable ({e}). Fallback mode.")
    finally:
        if conn:
            cur.close()
            conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_relational_database()
    yield

app = FastAPI(title="TrialGuard Underwriting Engine", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# GRAPH ROUTER SCHEMATICS
builder = StateGraph(TrialState)

builder.add_node("ROUTER", ingestion_router_node)
builder.add_node("cognitive_worker", cognitive_extractor_node)
builder.add_node("supervisor", managing_director_supervisor_node)
builder.add_node("medical_worker", medical_researcher_node)
builder.add_node("fintech_worker", fintech_underwriter_node)
builder.add_node("risk_worker", risk_grader_node)
builder.add_node("circuit_breaker", medical_circuit_breaker_node)
builder.add_node("human_breakpoint_barrier", lambda state: {})

builder.set_entry_point("ROUTER")

def runtime_entry_switchboard(state: TrialState):
    path = state.get("ingestion_routing_path", "HYBRID")
    if path == "COGNITIVE":
        return "to_cognitive"
    elif path == "MALFORMED":
        return "to_end"
    else:
        return "to_standard_pipeline"

builder.add_conditional_edges("ROUTER", runtime_entry_switchboard, {
    "to_cognitive": "cognitive_worker",
    "to_standard_pipeline": "supervisor",
    "to_end": "human_breakpoint_barrier"
})

builder.add_edge("cognitive_worker", "supervisor")

def switchboard_router(state: TrialState):
    target = state["next_step"]
    if target == "RESEARCHER": return "to_med"
    elif target == "FINTECH_AUDITOR": return "to_fin"
    elif target == "RISK_GRADER": return "to_risk"
    else: return "to_breaker"

builder.add_conditional_edges("supervisor", switchboard_router, {
    "to_med": "medical_worker", 
    "to_fin": "fintech_worker", 
    "to_risk": "risk_worker", 
    "to_breaker": "circuit_breaker",
})

builder.add_edge("medical_worker", "supervisor")
builder.add_edge("fintech_worker", "supervisor")
builder.add_edge("risk_worker", "supervisor")
builder.add_edge("circuit_breaker", "human_breakpoint_barrier")
builder.add_edge("human_breakpoint_barrier", END)

memory_db = MemorySaver()
pipeline = builder.compile(checkpointer=memory_db, interrupt_before=["human_breakpoint_barrier"])

def query_aggregated_metrics() -> MetricsResponse:
    conn = None
    try:
        conn = psycopg2.connect(DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(payout_amount), SUM(dispute_amount) FROM corporate_ledger;")
        row = cur.fetchone()
        cur.execute("SELECT SUM(claim_amount) FROM corporate_ledger WHERE status = 'ESCALATE';")
        escrow_row = cur.fetchone()
        
        return MetricsResponse(
            total_claims_processed=row[0] or 0, total_payouts_authorized=float(row[1] or 0.0),
            total_capital_leakage_prevented=float(row[2] or 0.0), active_disputed_escrow=float(escrow_row[0] or 0.0)
        )
    except Exception:
        return MetricsResponse(total_claims_processed=0, total_payouts_authorized=0.0, total_capital_leakage_prevented=0.0, active_disputed_escrow=0.0)
    finally:
        if conn:
            cur.close()
            conn.close()

@app.get("/")
async def root_health_check():
    return {
        "status": "ONLINE",
        "engine": "TrialGuard Multi-Agent Underwriting Ledger System",
        "version": "2.0"
    }

@app.get("/api/claims/metrics", response_model=MetricsResponse)
async def get_live_metrics():
    return query_aggregated_metrics()

@app.get("/api/claims/history")
async def get_ledger_history():
    conn = None
    try:
        conn = psycopg2.connect(DB_PARAMS)
        cur = conn.cursor()
        cur.execute("""
            SELECT thread_id, trial_id, patient_id, patient_name, patient_sex, patient_age, patient_cohort, 
                patient_enrollment, patient_incident, clinical_history, invoice_isolation, invoice_labor, 
                invoice_medication, claim_amount, payout_amount, dispute_amount, matched_rule, matched_clause, 
                confidence_score, status, TO_CHAR(timestamp_updated, 'DD-MM-YYYY HH24:MI:SS'), raw_clause_text, system_verdict, raw_rule_text
            FROM corporate_ledger ORDER BY id DESC LIMIT 10;
        """)
        rows = cur.fetchall()
        return [
            {
                "thread_id": r[0], "trial_id": r[1], "patient_id": r[2], "patient_name": r[3], "patient_sex": r[4], "patient_age": r[5],
                "patient_cohort": r[6], "patient_enrollment": r[7], "patient_incident": r[8], "clinical_history_summary": r[9],
                "invoice_isolation_fees": float(r[10]), "invoice_labor_fees": float(r[11]), "invoice_medication_fees": float(r[12]),
                "claim_amount": float(r[13]), "payout_amount": float(r[14]), "dispute_amount": float(r[15]),
                "matched_rule_id": r[16], "matched_clause_id": r[17], "rag_confidence_percentage": float(r[18]),
                "status": r[19], "time_approved": r[20],
                "raw_clause_text": r[21] or "No clause text found.",
                "system_verdict": r[22] or "No automated audit remarks logged.", 
                "raw_rule_text": r[23] or "No rule context text found." 
            } for r in rows
        ]
    except Exception: 
        return []
    finally:
        if conn:
            cur.close()
            conn.close()

@app.post("/api/claims/ingest", response_model=EnterpriseResponse)
async def ingest_claim(payload: ClaimIngestRequest, thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        history_summary = "\n".join([f"[{h.date} | {h.severity} | {h.condition}]" for h in payload.patient_profile.history_logs])
        prior_events_count = len(payload.patient_profile.history_logs)
        
        initial_state = {
            "trial_id": payload.trial_id,
            "patient_id": payload.patient_profile.subject_id,
            "procedure_notes": payload.procedure,
            "claim_amount": float(payload.invoice.isolation_ward_fees + payload.invoice.compounding_labor_fees + payload.invoice.auxiliary_medication_fees),
            "patient_name": payload.patient_profile.full_name,
            "patient_sex": payload.patient_profile.biological_sex,
            "patient_age": payload.patient_profile.age,
            "patient_cohort": payload.patient_profile.trial_cohort,
            "patient_enrollment": payload.patient_profile.enrollment_date,
            "patient_incident_date": payload.patient_profile.event_incident_date,
            "patient_telemetry": payload.patient_profile.telemetry_status,
            "clinical_history_summary": history_summary,
            "patient_prior_events": f"{prior_events_count} prior events",
            "invoice_isolation_fees": payload.invoice.isolation_ward_fees,
            "invoice_labor_fees": payload.invoice.compounding_labor_fees,
            "invoice_medication_fees": payload.invoice.auxiliary_medication_fees,
            "authorized_payout": 0.0,
            "escrow_dispute_amount": 0.0,
            "medical_context": "",
            "financial_context": "",
            "triage_verdict": "",
            "matched_rule_id": "PENDING",
            "matched_clause_id": "PENDING",
            "raw_rule_text": "", 
            "raw_clause_text": "",
            "rag_confidence_percentage": 0.0,
            "medical_checked": False,
            "financial_checked": False,
            "triage_checked": False,
            "breaker_checked": False,
            "approval_status": "PENDING",
            "next_step": ""
        }
        pipeline.invoke(initial_state, config=config)
        state_view = pipeline.get_state(config).values
        
        # 🛡️ SYSTEM FIX: Secure string retrieval targets dynamically
        res_rule_text = state_view.get("raw_rule_text") or "Standard baseline protocol context."
        res_clause_text = state_view.get("raw_clause_text") or "Standard contract baseline boundaries."

        return EnterpriseResponse(
            thread_id=thread_id, status="PAUSED", claim_amount=state_view.get("claim_amount", 0.0), authorized_payout=state_view.get("authorized_payout", 0.0), escrow_dispute=state_view.get("escrow_dispute_amount", 0.0), verdict=state_view.get("triage_verdict", ""),
            patient_name=state_view.get("patient_name", ""), patient_sex=state_view.get("patient_sex", ""), patient_age=state_view.get("patient_age", 0), patient_cohort=state_view.get("patient_cohort", ""),
            patient_enrollment=state_view.get("patient_enrollment", ""), patient_incident_date=state_view.get("patient_incident_date", ""), patient_telemetry=state_view.get("patient_telemetry", ""), clinical_history_summary=state_view.get("clinical_history_summary", ""),
            invoice_isolation_fees=state_view.get("invoice_isolation_fees", 0.0), invoice_labor_fees=state_view.get("invoice_labor_fees", 0.0), invoice_medication_fees=state_view.get("invoice_medication_fees", 0.0),
            matched_rule_id=state_view.get("matched_rule_id", "REG-NONE"), 
            matched_clause_id=state_view.get("matched_clause_id", "POLICY-UNKNOWN"), 
            raw_rule_text=res_rule_text,
            raw_clause_text=res_clause_text, 
            rag_confidence_percentage=state_view.get("rag_confidence_percentage", 94.5), 
            metrics=query_aggregated_metrics()
        )
    except Exception as e:
        print(f"INGESTION SERVER ERROR Traceback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/claims/action", response_model=EnterpriseResponse)
async def signoff_claim(payload: HumanSignoffRequest, thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # 🌟 CAPTURE THE VALID FINTECH NUMBERS FIRST BEFORE THE GRAPH EXITS
        state_view = pipeline.get_state(config).values
        
        if payload.action == "ESCALATE":
            payout_final = 0.0
            dispute_final = float(state_view.get("claim_amount", 0.0))
        else:
            payout_final = float(state_view.get("authorized_payout", 0.0))
            dispute_final = float(state_view.get("escrow_dispute_amount", 0.0))
            
        # Extract metadata metrics safely
        m_rule_id = state_view.get("matched_rule_id") or "REG-NONE"
        m_clause_id = state_view.get("matched_clause_id") or "POLICY-UNKNOWN"
        response_rule_text = state_view.get("raw_rule_text") or "Standard baseline protocol context."
        response_clause_text = state_view.get("raw_clause_text") or "Standard contract baseline boundaries."
        response_verdict_text = state_view.get("triage_verdict", "")

        # Now update state and exit the graph safely
        pipeline.update_state(config, {"approval_status": payload.action}, as_node="human_breakpoint_barrier")
        pipeline.invoke(None, config=config)
        
        conn = None
        try:
            conn = psycopg2.connect(DB_PARAMS)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO corporate_ledger (
                    thread_id, trial_id, patient_id, patient_name, patient_sex, patient_age, patient_cohort, 
                    patient_enrollment, patient_incident, clinical_history, invoice_isolation, invoice_labor, 
                    invoice_medication, claim_amount, payout_amount, dispute_amount, matched_rule, matched_clause, 
                    confidence_score, status, timestamp_updated, raw_clause_text, system_verdict, raw_rule_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET 
                    status = EXCLUDED.status, 
                    payout_amount = EXCLUDED.payout_amount, 
                    dispute_amount = EXCLUDED.dispute_amount, 
                    timestamp_updated = CURRENT_TIMESTAMP;
            """, (
                thread_id, state_view.get("trial_id"), state_view.get("patient_id"), state_view.get("patient_name"), state_view.get("patient_sex"), state_view.get("patient_age"), state_view.get("patient_cohort"),
                state_view.get("patient_enrollment"), state_view.get("patient_incident_date"), state_view.get("clinical_history_summary"), state_view.get("invoice_isolation_fees"), state_view.get("invoice_labor_fees"),
                state_view.get("invoice_medication_fees"), state_view.get("claim_amount"), payout_final, dispute_final, m_rule_id, m_clause_id, state_view.get("rag_confidence_percentage", 94.5), payload.action, 
                response_clause_text, response_verdict_text, response_rule_text
            ))
            conn.commit()
        except Exception as db_err: 
            print(f"DATABASE WRITE FAILURE: {db_err}")
        finally:
            if conn:
                cur.close()
                conn.close()
            
        return EnterpriseResponse(
            thread_id=thread_id, status=f"CLOSED_{payload.action}", claim_amount=state_view.get("claim_amount", 0.0), authorized_payout=payout_final, escrow_dispute=dispute_final, verdict=response_verdict_text,
            patient_name=state_view.get("patient_name", ""), patient_sex=state_view.get("patient_sex", ""), patient_age=state_view.get("patient_age", 0), patient_cohort=state_view.get("patient_cohort", ""),
            patient_enrollment=state_view.get("patient_enrollment", ""), patient_incident_date=state_view.get("patient_incident_date", ""), patient_telemetry=state_view.get("patient_telemetry", "CLOSED"), clinical_history_summary=state_view.get("clinical_history_summary", ""),
            invoice_isolation_fees=state_view.get("invoice_isolation_fees", 0.0), invoice_labor_fees=state_view.get("invoice_labor_fees", 0.0), invoice_medication_fees=state_view.get("invoice_medication_fees", 0.0),
            matched_rule_id=m_rule_id, matched_clause_id=m_clause_id, 
            raw_rule_text=response_rule_text, raw_clause_text=response_clause_text,
            rag_confidence_percentage=state_view.get("rag_confidence_percentage", 94.5), metrics=query_aggregated_metrics()
        )
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/claims/purge/{thread_id}")
async def purge_ledger_record(thread_id: str, admin_signature: str):
    if admin_signature != "admin": raise HTTPException(status_code=403, detail="Denied")
    conn = None
    try:
        conn = psycopg2.connect(DB_PARAMS)
        cur = conn.cursor()
        cur.execute("DELETE FROM corporate_ledger WHERE thread_id = %s;", (thread_id,))
        conn.commit()
        return {"status": "SUCCESS"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # uvicorn.run(app, host="0.0.0.0", port=port) 
    uvicorn.run(app, host="127.0.0.1", port=port) 