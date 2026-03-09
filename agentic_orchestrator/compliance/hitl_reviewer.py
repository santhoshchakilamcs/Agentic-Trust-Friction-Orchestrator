"""
Human-in-the-Loop (HITL) Reviewer

Pauses the autonomous pipeline when a high-stakes decision requires
human approval. In production this would integrate with a review queue
(e.g., Slack, internal dashboard). Here we support:
  - Interactive CLI prompts (for demo mode)
  - Programmatic injection (for tests / automated pipelines)
"""

from typing import Callable, Optional

from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.compliance.pii_masker import PIIMasker
from agentic_orchestrator.config import ACTION_APPROVE, ACTION_BLOCK, ACTION_ESCALATE


# Type alias for a review callback
ReviewCallback = Callable[[TransactionState], str]  # returns "APPROVE" | "REJECT" | "OVERRIDE"


def _interactive_review(state: TransactionState) -> str:
    """CLI-based human review prompt (used in demo mode)."""
    masked = PIIMasker.mask_dict(state.to_dict())
    print("\n" + "=" * 60)
    print("🛑  HUMAN REVIEW REQUIRED")
    print("=" * 60)
    print(f"  Transaction : {state.txn_id}")
    print(f"  Amount      : ${state.amount_usd:.2f}")
    print(f"  Corridor    : {state.corridor}")
    print(f"  Risk Score  : {state.risk_score:.4f}")
    print(f"  Risk Action : {state.risk_action}")
    print(f"  Compliance  : {', '.join(state.compliance_flags) or 'None'}")
    if state.risk_reasoning:
        print(f"  Reasoning   : {state.risk_reasoning[:200]}")
    print("-" * 60)
    print("  Options:  [A]pprove  |  [R]eject  |  [O]verride (approve + note)")
    print("-" * 60)

    while True:
        choice = input("  Your decision > ").strip().upper()
        if choice in ("A", "APPROVE"):
            return "APPROVE"
        elif choice in ("R", "REJECT"):
            return "REJECT"
        elif choice in ("O", "OVERRIDE"):
            return "OVERRIDE"
        print("  ⚠️  Please enter A, R, or O.")


class HITLReviewer:
    """
    Manages human-in-the-loop review for escalated transactions.

    Parameters
    ----------
    review_callback : callable, optional
        A function that receives a TransactionState and returns one of
        "APPROVE", "REJECT", or "OVERRIDE". If None, falls back to
        interactive CLI input.
    enabled : bool
        If False, auto-approves all HITL requests (useful for batch runs).
    """

    def __init__(
        self,
        review_callback: Optional[ReviewCallback] = None,
        enabled: bool = True,
    ):
        self._callback = review_callback or _interactive_review
        self.enabled = enabled
        self.review_log: list[dict] = []

    def needs_review(self, state: TransactionState) -> bool:
        """Check whether this transaction requires human review."""
        return (
            state.compliance_override == ACTION_ESCALATE
            or state.hitl_required
        )

    def review(self, state: TransactionState) -> TransactionState:
        """
        Pause the pipeline and request human review.
        Updates state with the human's decision.
        """
        if not self.needs_review(state):
            return state

        state.hitl_required = True
        state.log("HITL", "⏸️  Pipeline paused — awaiting human review")

        if not self.enabled:
            # Auto-escalate when HITL is disabled (batch/test mode)
            state.hitl_decision = "APPROVE"
            state.hitl_reviewer_notes = "AUTO-APPROVED (HITL disabled)"
            state.log("HITL", "Auto-approved (HITL disabled for batch run)")
        else:
            decision = self._callback(state)
            state.hitl_decision = decision
            if decision == "OVERRIDE":
                notes = ""
                if self._callback is _interactive_review:
                    notes = input("  Override notes > ").strip()
                state.hitl_reviewer_notes = notes or "Human override applied"

        # Apply the human decision
        if state.hitl_decision == "APPROVE":
            state.final_action = ACTION_APPROVE
            state.final_reasoning = (
                f"HUMAN APPROVED: {state.hitl_reviewer_notes or 'Reviewer approved the transaction'}. "
                f"Original risk: {state.risk_action} (score {state.risk_score:.4f})"
            )
            state.log("HITL", "✅ Human APPROVED the transaction")

        elif state.hitl_decision == "REJECT":
            state.final_action = ACTION_BLOCK
            state.final_reasoning = (
                f"HUMAN REJECTED: Transaction blocked by reviewer. "
                f"Original risk: {state.risk_action} (score {state.risk_score:.4f})"
            )
            state.log("HITL", "🚫 Human REJECTED the transaction")

        elif state.hitl_decision == "OVERRIDE":
            state.final_action = ACTION_APPROVE
            state.final_reasoning = (
                f"HUMAN OVERRIDE: {state.hitl_reviewer_notes}. "
                f"Original risk: {state.risk_action} (score {state.risk_score:.4f})"
            )
            state.log("HITL", f"🔄 Human OVERRIDE: {state.hitl_reviewer_notes}")

        # Record in review log for audit trail
        self.review_log.append({
            "txn_id": state.txn_id,
            "decision": state.hitl_decision,
            "notes": state.hitl_reviewer_notes,
            "original_risk_action": state.risk_action,
            "original_risk_score": state.risk_score,
        })

        return state

