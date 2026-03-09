"""
Compliance Firewall: Hard-coded rules that OVERRIDE AI agent decisions.
Handles AML thresholds, sanctions screening, and HITL triggers.
"""

from agentic_orchestrator.config import (
    ACTION_BLOCK,
    ACTION_ESCALATE,
    AML_THRESHOLD_USD,
    SANCTIONS_COUNTRIES,
)
from agentic_orchestrator.memory.short_term import TransactionState


class ComplianceFirewall:
    """
    Deterministic safety layer. These rules are NON-NEGOTIABLE and
    override any AI agent recommendation.
    """

    def check(self, state: TransactionState) -> TransactionState:
        """Run all compliance checks. Sets compliance_override if triggered."""
        state.log("ComplianceFirewall", "Running compliance checks...")

        # 1. Sanctions screening
        if state.recipient_country in SANCTIONS_COUNTRIES:
            state.compliance_override = ACTION_BLOCK
            state.compliance_flags.append(f"BLOCKED: Sanctioned destination country ({state.recipient_country})")
            state.log("ComplianceFirewall", f"🚫 SANCTIONS HIT: {state.recipient_country}")
            return state

        # 2. AML threshold — mandatory human review
        if state.amount_usd >= AML_THRESHOLD_USD:
            state.compliance_override = ACTION_ESCALATE
            state.compliance_flags.append(
                f"AML threshold exceeded: ${state.amount_usd:.2f} >= ${AML_THRESHOLD_USD:.2f}"
            )
            state.log("ComplianceFirewall", f"⚠️ AML THRESHOLD: ${state.amount_usd:.2f} requires HITL review")
            # Don't return — allow other checks to run too

        # 3. Structuring detection (multiple transactions just below threshold)
        # This would normally check recent history; placeholder for now
        if state.amount_usd > AML_THRESHOLD_USD * 0.9 and state.is_new_recipient:
            if ACTION_ESCALATE not in (state.compliance_override or ""):
                state.compliance_flags.append("Potential structuring: high amount to new recipient")
                state.log("ComplianceFirewall", "⚠️ Potential structuring pattern detected")

        # 4. Velocity check placeholder
        # In production, this would check transaction count in last N hours

        if not state.compliance_flags:
            state.log("ComplianceFirewall", "✅ All compliance checks passed")

        return state

    def apply_override(self, state: TransactionState) -> TransactionState:
        """If compliance flagged an override, apply it to the final decision."""
        if state.compliance_override:
            state.final_action = state.compliance_override
            state.final_reasoning = (
                f"COMPLIANCE OVERRIDE: {'; '.join(state.compliance_flags)}. "
                f"Original agent recommendation ({state.risk_action}) was overridden."
            )
            state.log("ComplianceFirewall", f"Final decision overridden to: {state.compliance_override}")
        return state
