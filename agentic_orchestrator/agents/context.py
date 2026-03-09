"""
Context Agent: Retrieves user history and life-event context from long-term memory.
Adjusts risk based on whether the transaction fits a known behavioral pattern.
Uses Claude API for reasoning with heuristic fallback.
"""

import json
from agentic_orchestrator.agents.base import BaseAgent
from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.llm.client import call_claude_json, is_available

CONTEXT_SYSTEM = """You are a context analysis agent for Remitly, a cross-border payment company.
Given a transaction and the user's historical profile, determine if the transaction fits known patterns.

Analyze:
- Is the recipient known or new?
- Is the corridor (sending/receiving country pair) consistent with history?
- Are there life events (wedding, medical, education) that explain unusual amounts?
- Is the amount within the user's normal range?
- Is the device the registered one?

Respond ONLY with valid JSON:
{
  "score_adjustment": -0.15,  // float between -0.3 and 0.3 (negative = less risky)
  "findings": ["Known recipient: Maria (trust +)", "Consistent corridor: US-PH"],
  "life_events": ["wedding preparation"],  // or empty list
  "reasoning": "brief explanation"
}"""


class ContextAgent(BaseAgent):
    name = "ContextAgent"

    def __init__(self, memory: LongTermMemory):
        self.memory = memory

    def run(self, state: TransactionState) -> TransactionState:
        state.log(self.name, "Searching user context and history...")

        # Retrieve context from vector memory (always needed)
        query = (
            f"Transaction of ${state.amount_usd:.2f} to {state.recipient_name} "
            f"in {state.recipient_country} via {state.corridor}"
        )
        context = self.memory.query_user_context(state.user_id, query)

        if not context["found"]:
            state.log(self.name, "No context found for this user.")
            state.context_score_adjustment = 0.0
            state.context_findings = ["No historical context available"]
            return state

        profile_data = context["metadata"][0] if context["metadata"] else {}
        profile_doc = context["documents"][0] if context["documents"] else ""

        if is_available():
            result = self._run_llm(state, profile_data, profile_doc)
            if result is not None:
                return result

        state.log(self.name, "Using heuristic fallback...")
        return self._run_heuristic(state, profile_data, profile_doc)

    def _run_llm(self, state: TransactionState, profile_data: dict, profile_doc: str):
        user_prompt = (
            f"Transaction details:\n"
            f"- Amount: ${state.amount_usd:.2f}\n"
            f"- Recipient: {state.recipient_name} (new: {state.is_new_recipient})\n"
            f"- Country: {state.recipient_country}, Corridor: {state.corridor}\n"
            f"- Device: {state.device_id}\n"
            f"- Investigator score: {state.investigator_score}\n"
            f"- Investigator flags: {state.investigator_flags}\n\n"
            f"User profile data: {json.dumps(profile_data, default=str)}\n"
            f"User profile document: {profile_doc}"
        )

        response = call_claude_json(CONTEXT_SYSTEM, user_prompt)
        if response is None:
            return None

        adjustment = float(response.get("score_adjustment", 0.0))
        adjustment = max(-0.3, min(0.3, adjustment))
        findings = response.get("findings", [])
        life_events = response.get("life_events", [])
        reasoning = response.get("reasoning", "")

        state.context_score_adjustment = round(adjustment, 4)
        state.context_findings = findings
        state.context_life_events = life_events

        state.log(self.name, f"[Claude] {reasoning}")
        state.log(self.name, f"Context analysis complete. Score adjustment: {state.context_score_adjustment:+.4f}")
        for f in findings:
            state.log(self.name, f"  → {f}")
        return state

    def _run_heuristic(self, state: TransactionState, profile_data: dict, profile_doc: str) -> TransactionState:
        """Original heuristic-based context analysis (fallback)."""
        findings = []
        adjustment = 0.0

        known_recipients_raw = profile_data.get("recipients", "[]")
        try:
            known_recipients = json.loads(known_recipients_raw)
        except (json.JSONDecodeError, TypeError):
            known_recipients = []

        if state.recipient_name in known_recipients:
            adjustment -= 0.15
            findings.append(f"Known recipient: {state.recipient_name} (trust +)")
        elif state.is_new_recipient:
            findings.append(f"First-time recipient: {state.recipient_name}")

        typical_corridor = profile_data.get("corridor", "")
        if state.corridor == typical_corridor:
            adjustment -= 0.05
            findings.append(f"Consistent corridor: {state.corridor}")
        else:
            adjustment += 0.05
            findings.append(f"Unusual corridor: {state.corridor} (typical: {typical_corridor})")

        life_events = []
        if "Life events:" in profile_doc:
            events_text = profile_doc.split("Life events:")[1].split(".")[0].strip()
            if events_text and events_text != "No known life events":
                life_events.append(events_text)
                adjustment -= 0.10
                findings.append(f"Life event context: {events_text}")

        typical_amount = profile_data.get("typical_amount", 0)
        if typical_amount > 0:
            ratio = state.amount_usd / typical_amount
            if ratio <= 1.5:
                adjustment -= 0.05
                findings.append(f"Amount within normal range ({ratio:.1f}x typical)")
            elif ratio <= 3.0:
                findings.append(f"Amount elevated ({ratio:.1f}x typical)")
            else:
                findings.append(f"Amount significantly high ({ratio:.1f}x typical)")

        known_device = profile_data.get("device_id", "")
        if known_device and state.device_id == known_device:
            adjustment -= 0.05
            findings.append("Using registered device (trust +)")

        state.context_score_adjustment = round(max(-0.3, min(0.3, adjustment)), 4)
        state.context_findings = findings
        state.context_life_events = life_events

        state.log(self.name, f"Context analysis complete. Score adjustment: {state.context_score_adjustment:+.4f}")
        for f in findings:
            state.log(self.name, f"  → {f}")
        return state

