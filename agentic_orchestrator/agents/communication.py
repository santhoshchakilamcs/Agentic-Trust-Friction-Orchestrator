"""
Communication Agent: Generates personalized, empathetic verification messages
when a transaction is flagged for challenge (instead of a blunt block).
Uses Claude API for natural message generation with heuristic fallback.
"""

from agentic_orchestrator.agents.base import BaseAgent
from agentic_orchestrator.config import ACTION_CHALLENGE, ACTION_ESCALATE
from agentic_orchestrator.llm.client import call_claude, is_available
from agentic_orchestrator.memory.short_term import TransactionState

COMMUNICATION_SYSTEM = """You are a customer communication agent for Remitly, a cross-border payment company.
Generate a SHORT, empathetic, friendly verification message for the customer.

Rules:
- Be warm and personal, NOT robotic or threatening
- Acknowledge the transaction details naturally
- If there are life events, reference them sensitively
- For CHALLENGE: ask for quick 2FA verification
- For ESCALATE: explain a brief review is needed (1-2 hours)
- Keep it under 3 sentences
- Include one relevant emoji at the end (🔒 for challenge, 🛡️ for escalation)
- Do NOT mention fraud, suspicion, or risk scores
- Do NOT wrap in quotes

Respond with ONLY the message text, nothing else."""


class CommunicationAgent(BaseAgent):
    name = "CommunicationAgent"

    def run(self, state: TransactionState) -> TransactionState:
        if state.risk_action not in (ACTION_CHALLENGE, ACTION_ESCALATE):
            state.log(self.name, "No challenge needed — skipping communication.")
            return state

        state.log(self.name, "Generating personalized verification message...")

        if is_available():
            message = self._run_llm(state)
            if message is not None:
                state.challenge_message = message
                state.log(self.name, f"[Claude] Message: {message}")
                return state

        state.log(self.name, "Using heuristic fallback...")
        if state.risk_action == ACTION_CHALLENGE:
            message = self._generate_challenge_message(state)
        else:
            message = self._generate_escalation_message(state)

        state.challenge_message = message
        state.log(self.name, f"Message: {message}")
        return state

    def _run_llm(self, state: TransactionState):
        user_prompt = (
            f"Action type: {state.risk_action}\n"
            f"Amount: ${state.amount_usd:.2f}\n"
            f"Recipient: {state.recipient_name} in {state.recipient_country}\n"
            f"Is new recipient: {state.is_new_recipient}\n"
            f"Life events: {state.context_life_events}\n"
            f"Context findings: {state.context_findings}\n"
            f"Compliance flags: {state.compliance_flags}"
        )
        return call_claude(COMMUNICATION_SYSTEM, user_prompt, max_tokens=256)

    def _generate_challenge_message(self, state: TransactionState) -> str:
        """Generate a friendly 2FA / soft-challenge message (fallback)."""
        context_hint = ""
        if state.context_life_events:
            context_hint = " We noticed this might be related to a special occasion. "
        elif state.is_new_recipient:
            context_hint = (
                f" Since this is your first transfer to {state.recipient_name}, "
                "we'd like to make sure everything is correct. "
            )

        return (
            f"Hi! We're processing your ${state.amount_usd:.2f} transfer "
            f"to {state.recipient_name} in {state.recipient_country}. "
            f"{context_hint}"
            f"For your security, please confirm this transaction with a quick verification. "
            f"This only takes a moment. 🔒"
        )

    def _generate_escalation_message(self, state: TransactionState) -> str:
        """Generate a message for transactions requiring human review (fallback)."""
        reasons = []
        if state.compliance_flags:
            reasons.extend(state.compliance_flags)
        elif state.investigator_flags:
            reasons.append("unusual activity patterns")

        reason_text = ", ".join(reasons) if reasons else "security review"

        return (
            f"Hi! Your ${state.amount_usd:.2f} transfer to {state.recipient_name} "
            f"requires a brief review due to {reason_text}. "
            f"Our team will review this shortly — typically within 1-2 hours. "
            f"We appreciate your patience and want to keep your account safe. 🛡️"
        )
