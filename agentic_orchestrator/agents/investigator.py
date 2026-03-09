"""
Investigator Agent: Analyzes transaction features to produce a raw risk signal.
Uses Claude API for reasoning with heuristic fallback.
"""

from agentic_orchestrator.agents.base import BaseAgent
from agentic_orchestrator.config import FEATURE_WEIGHTS, SANCTIONS_COUNTRIES
from agentic_orchestrator.llm.client import call_claude_json, is_available
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.memory.short_term import TransactionState

INVESTIGATOR_SYSTEM = """You are a fraud investigation agent for Remitly, a cross-border payment company.
Analyze the transaction and user profile data provided. Assess risk across these dimensions:
- amount_anomaly (0.0-1.0): Is the amount unusual vs. the user's typical pattern?
- new_recipient (0.0-1.0): Is this a first-time recipient? (0.6 if yes, 0.0 if no)
- device_change (0.0-1.0): Is the device different from the registered one?
- ip_velocity (0.0-1.0): Any IP-related concerns?
- country_risk (0.0-1.0): Is the destination country high-risk or sanctioned? (1.0 if sanctioned)
- time_anomaly (0.0-1.0): Is the transaction at an unusual time?
- frequency_anomaly (0.0-1.0): Any unusual frequency patterns?

Respond ONLY with valid JSON:
{
  "scores": {"amount_anomaly": 0.0, "new_recipient": 0.0, ...},
  "flags": ["list of human-readable risk flags"],
  "reasoning": "brief explanation of your analysis"
}"""


class InvestigatorAgent(BaseAgent):
    name = "InvestigatorAgent"

    def __init__(self, memory: LongTermMemory):
        self.memory = memory

    def run(self, state: TransactionState) -> TransactionState:
        state.log(self.name, "Starting transaction investigation...")

        profile = self.memory.get_user_profile(state.user_id)
        profile_info = ""
        if profile and profile["metadata"]:
            meta = profile["metadata"]
            profile_info = (
                f"User profile: typical_amount=${meta.get('typical_amount', 'unknown')}, "
                f"typical_corridor={meta.get('corridor', 'unknown')}, "
                f"registered_device={meta.get('device_id', 'unknown')}, "
                f"known_recipients={meta.get('recipients', '[]')}"
            )

        if is_available():
            result = self._run_llm(state, profile_info)
            if result is not None:
                return result

        state.log(self.name, "Using heuristic fallback...")
        return self._run_heuristic(state, profile)

    def _run_llm(self, state: TransactionState, profile_info: str):
        user_prompt = (
            f"Transaction details:\n"
            f"- Amount: ${state.amount_usd:.2f}\n"
            f"- Recipient: {state.recipient_name} in {state.recipient_country}\n"
            f"- Corridor: {state.corridor}\n"
            f"- Device: {state.device_id}\n"
            f"- IP: {state.ip_address}\n"
            f"- Time: {state.timestamp}\n"
            f"- New recipient: {state.is_new_recipient}\n\n"
            f"{profile_info}\n\n"
            f"Sanctioned countries: {', '.join(SANCTIONS_COUNTRIES)}"
        )

        response = call_claude_json(INVESTIGATOR_SYSTEM, user_prompt)
        if response is None:
            return None

        scores = response.get("scores", {})
        flags = response.get("flags", [])
        reasoning = response.get("reasoning", "")

        for key in FEATURE_WEIGHTS:
            scores.setdefault(key, 0.0)
            scores[key] = max(0.0, min(1.0, float(scores[key])))

        weighted_score = sum(scores.get(k, 0) * w for k, w in FEATURE_WEIGHTS.items())

        state.investigator_score = round(min(1.0, weighted_score), 4)
        state.investigator_flags = flags
        state.investigator_details = scores

        state.log(self.name, f"[Claude] {reasoning}")
        state.log(self.name, f"Investigation complete. Raw risk score: {state.investigator_score:.4f}")
        if flags:
            state.log(self.name, f"Flags raised: {', '.join(flags)}")
        return state

    def _run_heuristic(self, state: TransactionState, profile) -> TransactionState:
        """Original heuristic-based investigation (fallback)."""
        scores = {}
        flags = []

        if profile and profile["metadata"]:
            typical = profile["metadata"].get("typical_amount", 300)
            ratio = state.amount_usd / max(typical, 1)
            if ratio > 3.0:
                scores["amount_anomaly"] = min(1.0, ratio / 10.0)
                flags.append(f"Amount ${state.amount_usd:.2f} is {ratio:.1f}x typical ${typical:.2f}")
            elif ratio > 1.5:
                scores["amount_anomaly"] = 0.3
            else:
                scores["amount_anomaly"] = 0.0
        else:
            scores["amount_anomaly"] = 0.5
            flags.append("No user profile found in memory")

        if state.is_new_recipient:
            scores["new_recipient"] = 0.6
            flags.append(f"New recipient: {state.recipient_name}")
        else:
            scores["new_recipient"] = 0.0

        if profile and profile["metadata"]:
            known_device = profile["metadata"].get("device_id", "")
            if state.device_id != known_device:
                scores["device_change"] = 0.8
                flags.append(f"Unknown device {state.device_id} (expected {known_device})")
            else:
                scores["device_change"] = 0.0
                scores["ip_velocity"] = 0.0
        else:
            scores["device_change"] = 0.5
            scores["ip_velocity"] = 0.5

        if state.recipient_country in SANCTIONS_COUNTRIES:
            scores["country_risk"] = 1.0
            flags.append(f"Sanctioned destination: {state.recipient_country}")
        else:
            scores["country_risk"] = 0.0

        if state.timestamp:
            try:
                from datetime import datetime

                ts = datetime.fromisoformat(state.timestamp)
                hour = ts.hour
                if hour < 5 or hour > 23:
                    scores["time_anomaly"] = 0.4
                    flags.append(f"Unusual hour: {hour}:00")
                else:
                    scores["time_anomaly"] = 0.0
            except Exception:
                scores["time_anomaly"] = 0.1

        scores["frequency_anomaly"] = 0.0

        for key in FEATURE_WEIGHTS:
            scores.setdefault(key, 0.0)

        weighted_score = sum(scores.get(k, 0) * w for k, w in FEATURE_WEIGHTS.items())

        state.investigator_score = round(min(1.0, weighted_score), 4)
        state.investigator_flags = flags
        state.investigator_details = scores

        state.log(self.name, f"Investigation complete. Raw risk score: {state.investigator_score:.4f}")
        if flags:
            state.log(self.name, f"Flags raised: {', '.join(flags)}")
        return state
