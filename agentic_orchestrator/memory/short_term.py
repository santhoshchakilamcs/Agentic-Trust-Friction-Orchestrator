"""
Short-Term Memory: Session state for the current transaction being processed.
Stores intermediate agent outputs, current context, and orchestration state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TransactionState:
    """Mutable state that flows through the orchestrator graph."""

    # --- Input ---
    txn_id: str = ""
    user_id: str = ""
    amount_usd: float = 0.0
    recipient_name: str = ""
    recipient_country: str = ""
    corridor: str = ""
    device_id: str = ""
    ip_address: str = ""
    timestamp: str = ""
    is_new_recipient: bool = False

    # --- Agent Outputs ---
    investigator_score: float = 0.0
    investigator_flags: List[str] = field(default_factory=list)
    investigator_details: Dict[str, Any] = field(default_factory=dict)

    context_score_adjustment: float = 0.0
    context_findings: List[str] = field(default_factory=list)
    context_life_events: List[str] = field(default_factory=list)

    risk_score: float = 0.0
    risk_action: str = ""
    risk_reasoning: str = ""

    # --- Compliance ---
    compliance_override: Optional[str] = None
    compliance_flags: List[str] = field(default_factory=list)
    pii_masked: bool = False

    # --- Communication ---
    challenge_message: str = ""

    # --- HITL (Human-in-the-Loop) ---
    hitl_required: bool = False
    hitl_decision: Optional[str] = None  # "APPROVE", "REJECT", "OVERRIDE"
    hitl_reviewer_notes: str = ""

    # --- LLM-as-a-Judge ---
    judge_score: Optional[float] = None  # 0.0–1.0 quality grade
    judge_reasoning_grade: str = ""
    judge_feedback: str = ""

    # --- Final Decision ---
    final_action: str = ""
    final_reasoning: str = ""
    processing_log: List[str] = field(default_factory=list)

    def log(self, agent: str, message: str):
        self.processing_log.append(f"[{agent}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
