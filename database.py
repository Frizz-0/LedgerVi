# database.py
import ollama
import chromadb

class TrialGuardKnowledgeBase:
    def __init__(self):
        # Initialize the persistent local client layout configuration
        self.chroma_client = chromadb.Client()
        
        # Isolate medical regulations from financial contracts cleanly
        self.medical_collection = self.chroma_client.get_or_create_collection(name="medical_regs")
        self.financial_collection = self.chroma_client.get_or_create_collection(name="financial_contracts")
        
        self._seed_knowledge_base()

    def _seed_knowledge_base(self):
        print("📦 Seeding TrialGuard Enterprise Knowledge Bases with Expanded Rulesets...")
        
        # 1. Comprehensive Medical/Regulatory Standards Collection Array
        med_rules = {
            "med_802": "REG-802: Severe expanding skin lesions, acute urticaria, or systemic anaphylaxis during trial phases require high-priority specialist escalation and immediate isolation protocols.",
            "med_915": "REG-915: Administration of unapproved auxiliary macrolide compounds or non-protocol antibiotics requires a mandatory 24-hour continuous telemetry monitoring log.",
            "med_440": "REG-440: Any subject under the age threshold of 18 experiencing a Grade 3 adverse event must be automatically flagged for immediate clinical suspension and trial board oversight.",
            "med_112": "REG-112: Sudden baseline heart rate variances exceeding +/-30% following delivery of active max compounds require an immediate cardiac enzyme panel and emergency intensive care tracking.",
            "med_550": "REG-550: Experimental therapies administered outside primary trial sites during acute life-threatening episodes are legally permitted only if a prior multi-disciplinary site-board waiver is attached to the patient file record."
        }
        
        for key, doc in med_rules.items():
            res = ollama.embeddings(model="llama3.1", prompt=doc)
            self.medical_collection.add(ids=[key], embeddings=[res["embedding"]], documents=[doc])

        # 2. Comprehensive Financial/Underwriting Contract Clause Collection Array
        fin_clauses = {
            "fin_100k": "MAX-CAP-100K: Private ward isolation and auxiliary compound claims are strictly capped at a ceiling of £100,000 per adverse event cycle. Any overage amount is non-reimbursable.",
            "fin_303": "POLICY-303: Standard outpatient specialist consultation fees and routine secondary lab assays during dynamic trials are covered up to a maximum of £5,000 per fiscal day token.",
            "fin_705": "EXCL-705: Claims for auxiliary therapies or unvouched medications sourced from non-affiliated or un-audited pharmaceutical distribution nodes are strictly excluded from baseline underwriting coverage.",
            "fin_990": "LIMIT-990: Claims submitted by offshore or cross-border trial sites are subject to a maximum currency conversion protection limit of 5% variance; currency fluctuations exceeding this threshold must be absorbed by the local host research clinic.",
            "fin_450": "CAP-450: Intravenous compounding and specialized pharmacy formulation labor fees are strictly capped at a max-ceiling limit of £15,000 per subject admission block."
        }
        
        for key, doc in fin_clauses.items():
            res = ollama.embeddings(model="llama3.1", prompt=doc)
            self.financial_collection.add(ids=[key], embeddings=[res["embedding"]], documents=[doc])
            
        print("✅ Vector data streams securely indexed. 10 Core Enterprise Rules Live.")

    def query_medical(self, text: str) -> str:
        res = ollama.embeddings(model="llama3.1", prompt=text)
        results = self.medical_collection.query(query_embeddings=[res["embedding"]], n_results=1)
        return results["documents"][0][0] if results["documents"][0] else "No matching regulation found."

    def query_financial(self, text: str) -> tuple:
        res = ollama.embeddings(model="llama3.1", prompt=text)
        results = self.financial_collection.query(query_embeddings=[res["embedding"]], n_results=2)
        if results["documents"][0]:
            return (results["documents"][0][0], results["ids"][0][0]) if len(results["ids"][0]) > 0 else (results["documents"][0][0], "POLICY-UNKNOWN")
        return ("No matching contract clause found.", "POLICY-UNKNOWN")

# Singleton instance instantiation
trialguard_db = TrialGuardKnowledgeBase()