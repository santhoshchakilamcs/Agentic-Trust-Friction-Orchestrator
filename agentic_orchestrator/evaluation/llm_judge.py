"""
LLM-as-a-Judge Evaluator

Uses Claude to grade the *quality* of agentic reasoning,
not just the outcome. Evaluates:
  1. Reasoning Quality — did the agents produce coherent, evidence-based logic?
  2. Tool Usage — did the agents use available context (memory, compliance) correctly?
  3. Decision Appropriateness — does the final action match the evidence?
  4. Communication Quality — was the customer message empathetic & clear?

Falls back to deterministic heuristic rubric when API is unavailable.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.config import (
    ACTION_APPROVE, ACTION_BLOCK, ACTION_CHALLENGE, ACTION_ESCALATE,
    SANCTIONS_COUNTRIES, AML_THRESHOLD_USD,
)
from agentic_orchestrator.llm.client import call_claude_json, is_available

JUDGE_SYSTEM = """You are an expert evaluator grading the quality of an AI fraud detection system's reasoning.
Grade the following transaction processing on four dimensions (each 0.0-1.0):

1. reasoning_score: Did agents produce coherent, evidence-based logic?
2. tool_usage_score: Did agents use memory, compliance checks, and context correctly?
3. decision_score: Does the final action match the evidence? CRITICAL: sanctioned countries MUST be blocked, AML threshold transactions MUST trigger human review.
4. communication_score: Was the customer message (if any) empathetic and clear? (1.0 if no message was needed)

Respond ONLY with valid JSON:
{
  "reasoning_score": 0.9,
  "tool_usage_score": 0.85,
  "decision_score": 0.95,
  "communication_score": 0.9,
  "issues": ["list of any concerns"],
  "feedback": "overall assessment in 1-2 sentences"
}"""


@dataclass
class JudgeVerdict:
    """Structured evaluation from the LLM judge."""
    overall_score: float  # 0.0–1.0
    reasoning_score: float  # 0.0–1.0
    tool_usage_score: float  # 0.0–1.0
    decision_score: float  # 0.0–1.0
    communication_score: float  # 0.0–1.0
    grade: str  # A / B / C / D / F
    feedback: str  # Natural-language explanation
    issues: List[str] = field(default_factory=list)


class LLMJudge:
    """
    Evaluates the quality of an agentic decision pipeline.
    Uses Claude API with deterministic heuristic fallback.
    """

    def evaluate(self, state: TransactionState) -> JudgeVerdict:
        """
        Grade a completed transaction's agentic reasoning.
        Tries Claude API first, falls back to deterministic heuristic.
        """
        # Try LLM-based evaluation first
        if is_available():
            verdict = self._evaluate_with_llm(state)
            if verdict is not None:
                return verdict

        # Fallback to heuristic evaluation
        return self._evaluate_heuristic(state)

    def _evaluate_with_llm(self, state: TransactionState) -> Optional[JudgeVerdict]:
        """Use Claude to evaluate the agentic reasoning."""
        user_prompt = (
            f"Transaction: {state.txn_id}\n"
            f"Amount: ${state.amount_usd:.2f} to {state.recipient_name} "
            f"in {state.recipient_country}\n"
            f"Corridor: {state.corridor}\n"
            f"Sanctioned countries: {SANCTIONS_COUNTRIES}\n"
            f"AML threshold: ${AML_THRESHOLD_USD}\n\n"
            f"Investigator score: {state.investigator_score}\n"
            f"Investigator flags: {state.investigator_flags}\n"
            f"Context adjustment: {state.context_score_adjustment}\n"
            f"Context findings: {state.context_findings}\n"
            f"Final risk score: {state.risk_score}\n"
            f"Risk action: {state.risk_action}\n"
            f"Risk reasoning: {state.risk_reasoning}\n"
            f"Final action: {state.final_action}\n"
            f"Challenge message: {state.challenge_message}\n"
            f"HITL required: {state.hitl_required}\n"
            f"HITL decision: {state.hitl_decision}\n\n"
            f"Processing log:\n" + "\n".join(state.processing_log[-15:])
        )

        response = call_claude_json(JUDGE_SYSTEM, user_prompt)
        if response is None:
            return None

        reasoning_score = max(0.0, min(1.0, float(response.get("reasoning_score", 0.8))))
        tool_usage_score = max(0.0, min(1.0, float(response.get("tool_usage_score", 0.8))))
        decision_score = max(0.0, min(1.0, float(response.get("decision_score", 0.8))))
        communication_score = max(0.0, min(1.0, float(response.get("communication_score", 0.8))))
        issues = response.get("issues", [])
        feedback = response.get("feedback", "")

        overall = (
            reasoning_score * 0.30
            + tool_usage_score * 0.20
            + decision_score * 0.35
            + communication_score * 0.15
        )
        grade = self._score_to_grade(overall)

        verdict = JudgeVerdict(
            overall_score=round(overall, 4),
            reasoning_score=round(reasoning_score, 4),
            tool_usage_score=round(tool_usage_score, 4),
            decision_score=round(decision_score, 4),
            communication_score=round(communication_score, 4),
            grade=grade,
            feedback=feedback,
            issues=issues,
        )

        state.judge_score = verdict.overall_score
        state.judge_reasoning_grade = verdict.grade
        state.judge_feedback = verdict.feedback
        state.log("LLM-Judge", f"[Claude] Grade: {grade} ({overall:.2f}) — {feedback}")

        return verdict

    def _evaluate_heuristic(self, state: TransactionState) -> JudgeVerdict:
        """Deterministic heuristic evaluation (fallback)."""
        issues: List[str] = []

        reasoning_score = self._evaluate_reasoning(state, issues)
        tool_usage_score = self._evaluate_tool_usage(state, issues)
        decision_score = self._evaluate_decision(state, issues)
        communication_score = self._evaluate_communication(state, issues)

        overall = (
            reasoning_score * 0.30
            + tool_usage_score * 0.20
            + decision_score * 0.35
            + communication_score * 0.15
        )

        grade = self._score_to_grade(overall)
        feedback = self._generate_feedback(state, overall, issues)

        verdict = JudgeVerdict(
            overall_score=round(overall, 4),
            reasoning_score=round(reasoning_score, 4),
            tool_usage_score=round(tool_usage_score, 4),
            decision_score=round(decision_score, 4),
            communication_score=round(communication_score, 4),
            grade=grade,
            feedback=feedback,
            issues=issues,
        )

        state.judge_score = verdict.overall_score
        state.judge_reasoning_grade = verdict.grade
        state.judge_feedback = verdict.feedback
        state.log("LLM-Judge", f"Grade: {grade} ({overall:.2f}) — {len(issues)} issues found")

        return verdict

    # ------------------------------------------------------------------
    # Rubric evaluators
    # ------------------------------------------------------------------

    def _evaluate_reasoning(self, state: TransactionState, issues: list) -> float:
        score = 1.0
        if not state.risk_reasoning:
            score -= 0.5
            issues.append("No risk reasoning provided by agents")
        elif len(state.risk_reasoning) < 20:
            score -= 0.2
            issues.append("Risk reasoning is too brief")
        if not state.investigator_flags and state.investigator_score > 0.3:
            score -= 0.3
            issues.append("Investigator raised score without documenting flags")
        if len(state.processing_log) < 3:
            score -= 0.2
            issues.append("Sparse processing log — agents may have been skipped")
        return max(score, 0.0)

    def _evaluate_tool_usage(self, state: TransactionState, issues: list) -> float:
        score = 1.0
        # Check that context agent ran (should have findings)
        log_text = " ".join(state.processing_log)
        if "ContextAgent" not in log_text and "context" not in log_text.lower():
            score -= 0.4
            issues.append("Context agent does not appear in processing log")
        if "ComplianceFirewall" not in log_text:
            score -= 0.4
            issues.append("Compliance firewall does not appear in processing log")
        # Context findings should exist for non-blocked transactions
        if state.final_action != ACTION_BLOCK and not state.context_findings:
            score -= 0.2
            issues.append("No context findings recorded for non-blocked transaction")
        return max(score, 0.0)

    def _evaluate_decision(self, state: TransactionState, issues: list) -> float:
        score = 1.0
        # Sanctions must always be blocked
        if state.recipient_country in SANCTIONS_COUNTRIES:
            if state.final_action != ACTION_BLOCK:
                score = 0.0
                issues.append(f"CRITICAL: Sanctioned country {state.recipient_country} was NOT blocked")
                return score
        # AML threshold must escalate (or be human-reviewed)
        if state.amount_usd >= AML_THRESHOLD_USD:
            if state.final_action not in (ACTION_ESCALATE, ACTION_BLOCK, ACTION_APPROVE):
                score -= 0.3
                issues.append("AML threshold transaction has unexpected action")
            if not state.hitl_required and state.final_action == ACTION_APPROVE:
                score -= 0.3
                issues.append("AML transaction auto-approved without HITL review")
        # Low risk should be approved
        if state.risk_score < 0.2 and state.final_action != ACTION_APPROVE:
            if state.compliance_override is None:
                score -= 0.3
                issues.append("Low-risk transaction was not approved despite no compliance override")
        # High risk should not be auto-approved
        if state.risk_score > 0.7 and state.final_action == ACTION_APPROVE:
            if not state.hitl_decision:
                score -= 0.4
                issues.append("High-risk transaction auto-approved without human review")
        return max(score, 0.0)

    def _evaluate_communication(self, state: TransactionState, issues: list) -> float:
        score = 1.0
        if state.final_action == ACTION_CHALLENGE:
            if not state.challenge_message:
                score -= 0.5
                issues.append("Challenge action with no customer message")
            elif len(state.challenge_message) < 20:
                score -= 0.2
                issues.append("Challenge message is too brief")
        if state.final_action == ACTION_APPROVE:
            # No message needed — full score
            pass
        return max(score, 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.65:
            return "C"
        elif score >= 0.5:
            return "D"
        return "F"

    @staticmethod
    def _generate_feedback(state: TransactionState, score: float, issues: list) -> str:
        if not issues:
            return (
                f"Excellent agentic reasoning for {state.txn_id}. "
                f"All agents contributed evidence, compliance was checked, "
                f"and the final decision ({state.final_action}) is well-supported."
            )
        issue_list = "; ".join(issues[:3])
        return (
            f"Transaction {state.txn_id} scored {score:.2f}. "
            f"Issues: {issue_list}. "
            f"Final action: {state.final_action} (risk score: {state.risk_score:.4f})."
        )

