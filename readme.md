# LedgerVi — Clinical Trial Insurance Underwriting Engine

An enterprise-grade, multi-agent AI platform that automates the auditing, risk classification, and financial settlement of clinical trial adverse event insurance claims. Built on LangGraph stateful orchestration with a hybrid architecture that separates LLM-based reasoning from deterministic Python financial execution.

---

## Problem Statement

Clinical trial insurance claims present two structural failure modes that neither pure rule-based systems nor standard LLMs can solve alone.

**The Unstructured Data Problem.** Over 80% of clinical event data arrives as free-text medical narratives written by physicians. Traditional rule engines cannot parse human phrasing or distinguish a standard drug delivery from an emergency rescue compound deployment.

**The Deterministic Deficit of LLMs.** While large language models can read and reason over unstructured text, they are fundamentally non-deterministic and unreliable at executing precise financial arithmetic. Unconstrained LLMs routinely hallucinate numbers, misinterpret policy caps, and produce inconsistent results across identical inputs — an unacceptable failure mode in insurance underwriting.

LedgerVi solves both problems through architectural separation: LLMs handle reading and classification; Python handles all math.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                       INGESTION ROUTER                               │
│  Detects input shape: structured line items vs. unstructured text    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
           ┌────────────────┴────────────────┐
           │ Structured                      │ Unstructured
           ▼                                 ▼
  Skip extraction                 COGNITIVE EXTRACTOR
  (use raw figures)               (LLM parses text into
                                   structured line items)
           └────────────────┬────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 MANAGING DIRECTOR SUPERVISOR                         │
│  Stateful orchestrator. Routes workers in sequence.                  │
│  No peer-to-peer agent communication — all state flows through       │
│  the shared TrialState TypedDict.                                    │
└──────┬───────────────────────┬──────────────────────┬────────────────┘
       │                       │                      │
       ▼                       ▼                      ▼
MEDICAL RESEARCHER     FINTECH UNDERWRITER      RISK GRADER
Two-stage RAG          Hybrid audit engine      Composite threat
ChromaDB + LLM         LLM classifies rules     classification
reranker               Python executes math     CLEAR / HIGH_FINANCIAL
                                                / HIGH_MEDICAL_RISK
       └───────────────────────┴──────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   MEDICAL CIRCUIT BREAKER                            │
│  HIGH_MEDICAL_RISK  -->  authorized_payout = £0, full escrow hold    │
│  CLEAR / HIGH_FINANCIAL  -->  proceed to settlement                  │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│              HUMAN-IN-THE-LOOP BREAKPOINT                            │
│  Graph pauses here. Human reviewer approves or escalates via API.   │
│  Full state persisted to LangGraph MemorySaver checkpointer.        │
└──────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              PostgreSQL (corporate_ledger table)
```

---

## Two-Stage RAG Design

Naive single-stage vector retrieval causes false positives. A patient with mild skin redness matches "severe expanding skin lesions" by embedding proximity even when the clinical threshold is not met. LedgerVi uses a two-stage approach to eliminate this.

**Stage 1 — Broad Recall.** ChromaDB returns the top N candidate regulations or contract clauses by semantic similarity. This casts a wide net without making any decision.

**Stage 2 — LLM Reranker.** A 70B model reads the actual trigger conditions of each candidate against the exact clinical notes. It marks a regulation as applicable only when the notes explicitly document the required threshold — not when they merely resemble it. If no candidates pass, the pipeline returns `REG-NONE` cleanly. There is no fallback to the top hit.

This prevents semantic proximity from poisoning downstream risk classifications and financial decisions.

---

## Hybrid Financial Audit

The LLM is used exclusively to classify which policy rule applies to each billing line item and what the associated limit is. Python executes all arithmetic without any LLM involvement.

**Rule Types**

| Type | LLM Determines | Python Executes |
|---|---|---|
| `full_approval` | No clause applies | `approved = amount` |
| `cap` | Clause sets a ceiling; provides `limit` | `approved = min(amount, limit)` |
| `exclusion` | Fee explicitly not covered | `approved = 0` |
| `combined_cap` | Shared pool ceiling across multiple items | Pool budget allocated and decremented in order |

**Integrity check.** After all line items are processed, `authorized + disputed` must equal `total_claim` within £0.02. Any mismatch triggers `INTEGRITY-FAIL`, diverts the full claim to escrow, and flags the record.

**Matched Clause ID derivation.** The reported `matched_clause_id` is not simply the top ChromaDB hit. It is derived deterministically from the per-item dispute amounts — the clause responsible for the largest actual deduction becomes the authoritative matched clause. The displayed clause context is filtered to show only the documents referenced by the applied rule decisions.

---

## Dual LLM Strategy

| Model | Role | Justification |
|---|---|---|
| `llama-3.1-8b-instant` | Ingestion routing, invoice entity extraction | Low-stakes structural tasks; speed matters |
| `llama-3.3-70b-versatile` | Medical reranking, financial rule classification, risk grading | High-stakes reasoning requires full model capacity |

Both are served through Groq's inference API via LangChain's `ChatOpenAI` adapter.

---

## Medical Circuit Breaker

When the risk grader assigns `HIGH_MEDICAL_RISK`, a hard system override fires before the human checkpoint:

- `authorized_payout` is set to `£0.00`
- `escrow_dispute_amount` is set to the full gross claim
- All prior financial audit calculations are discarded

The circuit trips on any of: emergency protocol activation, immediate isolation or transfer, acute physiological instability (pyrexia, vital sign variance), explicit stabilization efforts, or Grade 2+ adverse events with emergency response. If both medical and financial risk flags are present simultaneously, `HIGH_MEDICAL_RISK` takes precedence.

---

## Knowledge Base

Stored in ChromaDB in-memory with two collections.

**Medical Regulations**

| ID | Trigger Condition |
|---|---|
| REG-802 | Severe skin lesions, urticaria, or systemic anaphylaxis — high-priority escalation and isolation required |
| REG-915 | Unapproved auxiliary macrolides or non-protocol antibiotics — 24-hour telemetry log required |
| REG-440 | Subject under 18 with Grade 3 adverse event — automatic clinical suspension and board oversight |
| REG-112 | Heart rate variance exceeding ±30% after active max compounds — cardiac enzyme panel and ICU tracking |
| REG-550 | Experimental therapies outside primary sites during life-threatening episodes — board waiver required |

**Financial Contract Clauses**

| ID | Rule |
|---|---|
| MAX-CAP-100K | Combined ceiling of £100,000 for private ward isolation and auxiliary compounds per adverse event cycle |
| POLICY-303 | Outpatient specialist and routine lab fees covered up to £5,000 per fiscal day |
| EXCL-705 | Auxiliary therapies from non-affiliated or un-audited pharmaceutical nodes — excluded from coverage |
| LIMIT-990 | Cross-border trial sites — currency variance beyond 5% threshold absorbed by host clinic |
| CAP-450 | IV compounding and pharmacy formulation labor — capped at £15,000 per admission block |

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI 0.136 |
| Agent Orchestration | LangGraph 1.2 |
| LLM Interface | LangChain OpenAI adapter (Groq backend) |
| Vector Store | ChromaDB 1.5 (in-memory) |
| Relational Database | PostgreSQL via psycopg2 |
| State Schema | Pydantic 2 + Python TypedDict |
| Runtime | Uvicorn / Python 3.12 |
| Frontend | Vanilla JS + Tailwind CSS |
| Containerization | Docker (Python 3.12-slim) |

---

## API Reference

### `POST /api/claims/ingest?thread_id={id}`

Submit a new claim. The pipeline runs to the human breakpoint and pauses, returning the full audit result for review.

```json
{
  "trial_id": "TRIAL-2026-001",
  "patient_profile": {
    "subject_id": "PT-8821",
    "full_name": "Jane Doe",
    "biological_sex": "Female",
    "age": 34,
    "trial_cohort": "Phase III Oncology",
    "enrollment_date": "2025-11-01",
    "event_incident_date": "2026-05-18",
    "telemetry_status": "ACTIVE",
    "history_logs": [
      { "date": "2026-03-10", "severity": "Grade 1", "condition": "Mild Nausea" }
    ]
  },
  "invoice": {
    "isolation_ward_fees": 90000.00,
    "compounding_labor_fees": 20000.00,
    "auxiliary_medication_fees": 15000.00
  },
  "procedure": "Clinical notes describing the adverse event..."
}
```

### `POST /api/claims/action?thread_id={id}`

Resume the paused pipeline with a human decision. Valid actions: `APPROVE`, `ESCALATE`.

```json
{ "action": "APPROVE" }
```

### `GET /api/claims/metrics`

Returns aggregate ledger statistics: total claims processed, total payouts authorized, capital leakage mitigated, active escrow holds.

### `GET /api/claims/history`

Returns the last 10 settled claims from the PostgreSQL ledger, including matched rule IDs, clause IDs, clause context text, and triage verdicts.

### `DELETE /api/claims/purge/{thread_id}?admin_signature=admin`

Removes a record from the ledger. Requires admin signature.

---

## Database Schema

All settled claims are written to the `corporate_ledger` table:

```sql
CREATE TABLE corporate_ledger (
    id                SERIAL PRIMARY KEY,
    thread_id         VARCHAR(100) UNIQUE,
    trial_id          VARCHAR(100),
    patient_id        VARCHAR(100),
    patient_name      VARCHAR(150),
    claim_amount      NUMERIC,
    payout_amount     NUMERIC,
    dispute_amount    NUMERIC,
    matched_rule      VARCHAR(50),
    matched_clause    VARCHAR(50),
    confidence_score  NUMERIC,
    status            VARCHAR(50),
    raw_clause_text   TEXT,
    raw_rule_text     TEXT,
    system_verdict    TEXT,
    timestamp_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Setup

**Prerequisites:** Python 3.12, a running PostgreSQL instance, and a Groq API key.

**1. Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**2. Configure environment variables** in a `.env` file at the project root:
```
GROQ_API_KEY=your_groq_api_key_here
Postgres_URL=postgresql://user:password@host:5432/dbname
```

**3. Start the server:**
```bash
cd backend
python server.py
```

ChromaDB seeds its knowledge base automatically on first startup. PostgreSQL schema is auto-initialized on the first request. The API is available at `http://localhost:8000`.

**4. Open the dashboard:**

Open `frontend/index.html` directly in a browser. No build step required.

**Docker:**
```bash
cd backend
docker build -t ledgervi .
docker run -p 8000:8000 --env-file ../.env ledgervi
```

---

## Project Structure

```
├── backend/
│   ├── server.py        — FastAPI application, LangGraph graph wiring, REST endpoints
│   ├── agents.py        — All agent node functions, RAG helpers, deterministic math engine
│   ├── database.py      — ChromaDB knowledge base: seeding, medical query, financial query
│   ├── engine_state.py  — TrialState TypedDict, Pydantic request/response models
│   ├── requirements.txt
│   └── dockerfile
└── frontend/
    └── index.html       — LedgerVi dashboard: metrics, claim submission, ledger history
```

---

## Key Design Decisions

**Why LangGraph over plain function calls?** LangGraph's stateful checkpointing natively supports human-in-the-loop breakpoints. The graph pauses mid-execution, persists its full state to a MemorySaver, and resumes from exactly the same point after a human decision — without re-running prior nodes.

**Why is the LLM not allowed to compute financial figures?** LLMs are non-deterministic. On identical inputs they may return different arithmetic results. Insurance underwriting requires reproducible, audit-compliant figures. The LLM is constrained to returning a classification label and a limit value only; Python computes all splits.

**Why two-stage RAG instead of single-stage?** Single-stage vector retrieval returns the most semantically similar document, not the most clinically applicable one. The reranker reads the actual trigger conditions and rejects candidates that do not meet the documented threshold, eliminating false regulatory matches.

**Why does the circuit breaker override the financial audit?** A patient in an active medical emergency cannot have their care constrained by a financial cap determination made before the clinical severity was known. The circuit breaker ensures HIGH_MEDICAL_RISK cases are fully frozen pending specialist review, regardless of the financial audit result.
