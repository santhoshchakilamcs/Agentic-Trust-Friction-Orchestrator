"""
Configuration and constants for the Agentic Risk & Retention Orchestrator.
"""

# --- AML / Compliance Thresholds ---
AML_THRESHOLD_USD = 3000.0  # Transactions above this require HITL review
SANCTIONS_COUNTRIES = {"KP", "IR", "SY", "CU"}  # ISO 3166-1 alpha-2 sanctioned destinations
HIGH_RISK_CORRIDORS = {"US-NG", "US-PK", "GB-BD"}  # Example high-risk corridors

# --- Risk Scoring ---
RISK_APPROVE_THRESHOLD = 0.3     # Below this -> auto-approve
RISK_CHALLENGE_THRESHOLD = 0.6   # Between approve and this -> soft challenge (2FA)
# Above challenge threshold -> escalate to human review

# --- Transaction Feature Weights (for Investigator Agent) ---
FEATURE_WEIGHTS = {
    "amount_anomaly": 0.25,
    "new_recipient": 0.20,
    "ip_velocity": 0.15,
    "device_change": 0.15,
    "country_risk": 0.10,
    "time_anomaly": 0.10,
    "frequency_anomaly": 0.05,
}

# --- Memory ---
CHROMA_COLLECTION_NAME = "user_context"
CHROMA_PERSIST_DIR = ".chroma_db"

# --- Supported Corridors ---
CORRIDORS = [
    "US-PH", "US-IN", "US-MX", "US-GT", "US-NG",
    "GB-PH", "GB-IN", "GB-PK", "GB-BD",
    "CA-PH", "CA-IN",
]

# --- Decision Actions ---
ACTION_APPROVE = "APPROVE"
ACTION_CHALLENGE = "CHALLENGE"
ACTION_ESCALATE = "ESCALATE_TO_HUMAN"
ACTION_BLOCK = "BLOCK"

# --- Random Seed for Reproducibility ---
RANDOM_SEED = 42

