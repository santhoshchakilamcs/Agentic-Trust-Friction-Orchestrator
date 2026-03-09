"""
Risk Scorer Agent: Synthesizes Investigator + Context outputs into a final score and action.
Uses Claude API for reasoning with heuristic fallback.
"""

from agentic_orchestrator.agents.base import BaseAgent
from agentic_orchestrator.config import (
    ACTION_APPROVE,
    ACTION_CHALLENGE,
    ACTION_ESCALATE,
    RISK_APPROVE_THRESHOLD,
    RISK_CHALLENGE_THRESHOLD,
)
from agentic_orchestrator.llm.client import call_claude_json, is_available
from agentic_orchestrator.memory.short_term import TransactionState

RISK_SCORER_SYSTEM = """You are a risk scoring agent for a cross-border payment platform.
You receive the investigator's raw risk score and flags, plus the context agent's findings and adjustment.
Synthesize these into a final risk decision.

Thresholds:
- APPROVE: final score < 0.3
- CHALLENGE: final score 0.3–0.7
- ESCALATE: final score > 0.7

IMPORTANT: The final_score MUST equal investigator_score + context_adjustment (clamped 0.0–1.0).
Do NOT invent a different score. Apply the thresholds above to determine the action.

Respond ONLY with valid JSON:
{
  "final_score": 0.15,
  "action": "APPROVE",
  "reasoning": "detailed explanation synthesizing investigator and context signals"
}"""


class RiskScorerAgent(BaseAgent):
    name = "RiskScorerAgent"

    def run(self, state: TransactionState) -> TransactionState:
        state.log(self.name, "Synthesizing risk assessment...")

        # Always compute the deterministic score
        raw_score = state.investigator_score
        adjustment = state.context_score_adjustment
        deterministic_score = max(0.0, min(1.0, raw_score + adjustment))

        if is_available():
            result = self._run_llm(state, deterministic_score)
            if result is not None:
                return result

        state.log(self.name, "Using heuristic fallback...")
        return self._run_heuristic(state, deterministic_score)

    def _run_llm(self, state: TransactionState, deterministic_score: float):
        user_prompt = (
            f"Investigator score: {state.investigator_score}\n"
            f"Investigator flags: {state.investigator_flags}\n"
            f"Context adjustment: {state.context_score_adjustment}\n"
            f"Context findings: {state.context_findings}\n"
            f"Life events: {state.context_life_events}\n"
            f"Computed final score: {deterministic_score:.4f}\n"
            f"Transaction: ${state.amount_usd:.2f} to {state.recipient_name} "
            f"in {state.recipient_country}"
        )

        response = call_claude_json(RISK_SCORER_SYSTEM, user_prompt)
        if response is None:
            return None

        # Use the deterministic score (don't let LLM override the math)
        final_score = deterministic_score
        reasoning = response.get("reasoning", "")

        if final_score < RISK_APPROVE_THRESHOLD:
            action = ACTION_APPROVE
        elif final_score < RISK_CHALLENGE_THRESHOLD:
            action = ACTION_CHALLENGE
        else:
            action = ACTION_ESCALATE

        state.risk_score = round(final_score, 4)
        state.risk_action = action
        state.risk_reasoning = reasoning

        state.log(self.name, f"[Claude] {reasoning}")
        state.log(self.name, f"Final risk score: {final_score:.4f} → Action: {action}")
        return state

    def _run_heuristic(self, state: TransactionState, final_score: float) -> TransactionState:
        """Original heuristic-based risk scoring (fallback)."""
        if final_score < RISK_APPROVE_THRESHOLD:
            action = ACTION_APPROVE
            reasoning = self._build_reasoning("LOW", state, final_score)
        elif final_score < RISK_CHALLENGE_THRESHOLD:
            action = ACTION_CHALLENGE
            reasoning = self._build_reasoning("MEDIUM", state, final_score)
        else:
            action = ACTION_ESCALATE
            reasoning = self._build_reasoning("HIGH", state, final_score)

        state.risk_score = round(final_score, 4)
        state.risk_action = action
        state.risk_reasoning = reasoning

        state.log(self.name, f"Final risk score: {final_score:.4f} → Action: {action}")
        state.log(self.name, f"Reasoning: {reasoning}")
        return state

    def _build_reasoning(self, level: str, state: TransactionState, score: float) -> str:
        parts = [f"Risk level: {level} (score: {score:.4f})."]

        if state.investigator_flags:
            parts.append(f"Investigation flags: {'; '.join(state.investigator_flags)}.")

        if state.context_findings:
            mitigating = [
                f for f in state.context_findings if "trust +" in f or "normal range" in f or "Life event" in f
            ]
            concerning = [f for f in state.context_findings if f not in mitigating]
            if mitigating:
                parts.append(f"Mitigating context: {'; '.join(mitigating)}.")
            if concerning:
                parts.append(f"Concerns: {'; '.join(concerning)}.")

        if state.context_life_events:
            parts.append("Life events support this transaction pattern.")

        return " ".join(parts)
