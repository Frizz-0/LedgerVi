# engine_state.py
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional

class InvoiceBreakdown(BaseModel):
    isolation_ward_fees: float = Field(..., example=90000.0)
    compounding_labor_fees: float = Field(..., example=20000.0)
    auxiliary_medication_fees: float = Field(..., example=15000.0)

class AdverseEventHistory(BaseModel):
    date: str = Field(..., example="2026-05-18")
    severity: str = Field(..., example="Grade 2")
    condition: str = Field(..., example="Localized Rash")

class PatientProfile(BaseModel):
    subject_id: str
    full_name: str
    biological_sex: str
    age: int
    trial_cohort: str
    enrollment_date: str
    event_incident_date: str
    telemetry_status: str
    history_logs: List[AdverseEventHistory]

class TrialState(TypedDict):
    trial_id: str
    patient_id: str
    procedure_notes: str
    ingestion_routing_path: str

    # Demographics
    patient_name: str
    patient_sex: str
    patient_age: int
    patient_cohort: str
    patient_enrollment: str
    patient_incident_date: str
    patient_telemetry: str
    clinical_history_summary: str
    patient_prior_events: str
    raw_clause_text: str
    raw_rule_text: str

    # Financial
    claim_amount: float
    authorized_payout: float
    escrow_dispute_amount: float
    invoice_isolation_fees: float
    invoice_labor_fees: float
    invoice_medication_fees: float

    # RAG Trust Metrics
    matched_rule_id: str
    matched_clause_id: str
    rag_confidence_percentage: float

    # Worker Context
    medical_context: str
    financial_context: str
    triage_verdict: str

    # Gate Flags
    medical_checked: bool
    financial_checked: bool
    triage_checked: bool
    breaker_checked: bool
    approval_status: str
    next_step: str

class ClaimIngestRequest(BaseModel):
    trial_id: str
    patient_profile: PatientProfile
    invoice: InvoiceBreakdown
    procedure: str

class HumanSignoffRequest(BaseModel):
    action: str

class MetricsResponse(BaseModel):
    total_claims_processed: int
    total_payouts_authorized: float
    total_capital_leakage_prevented: float
    active_disputed_escrow: float

class EnterpriseResponse(BaseModel):
    thread_id: str
    status: str
    claim_amount: float
    authorized_payout: float
    escrow_dispute: float
    verdict: str

    patient_name: str
    patient_sex: str
    patient_age: int
    patient_cohort: str
    patient_enrollment: str
    patient_incident_date: str
    patient_telemetry: str
    clinical_history_summary: str

    invoice_isolation_fees: float
    invoice_labor_fees: float
    invoice_medication_fees: float

    matched_rule_id: str
    matched_clause_id: str
    rag_confidence_percentage: float
    metrics: MetricsResponse
    raw_rule_text: str
    raw_clause_text: str
