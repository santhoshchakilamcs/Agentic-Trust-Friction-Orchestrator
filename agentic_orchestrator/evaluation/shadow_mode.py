"""
Shadow Mode Evaluator: Runs both the baseline and agentic systems side-by-side
and compares their decisions to measure improvement.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from agentic_orchestrator.config import ACTION_BLOCK, ACTION_CHALLENGE, ACTION_ESCALATE


@dataclass
class ShadowResult:
    txn_id: str
    is_fraud: bool
    ground_truth_label: str
    baseline_action: str
    baseline_score: float
    agentic_action: str
    agentic_score: float
    agentic_reasoning: str
    agreement: bool = False
    agentic_better: bool = False


@dataclass
class ShadowModeReport:
    total_transactions: int = 0
    total_fraud: int = 0
    total_legitimate: int = 0

    # Baseline metrics
    baseline_true_positives: int = 0  # Fraud correctly caught
    baseline_false_positives: int = 0  # Legit flagged as fraud
    baseline_true_negatives: int = 0  # Legit correctly approved
    baseline_false_negatives: int = 0  # Fraud missed

    # Agentic metrics
    agentic_true_positives: int = 0
    agentic_false_positives: int = 0
    agentic_true_negatives: int = 0
    agentic_false_negatives: int = 0

    agreement_count: int = 0
    agentic_improvements: int = 0  # Cases where agentic was better
    results: List[ShadowResult] = field(default_factory=list)

    def add_result(self, result: ShadowResult):
        self.results.append(result)
        self.total_transactions += 1

        is_flagged_baseline = result.baseline_action in (ACTION_CHALLENGE, ACTION_ESCALATE, ACTION_BLOCK)
        is_flagged_agentic = result.agentic_action in (ACTION_CHALLENGE, ACTION_ESCALATE, ACTION_BLOCK)

        if result.is_fraud:
            self.total_fraud += 1
            self.baseline_true_positives += int(is_flagged_baseline)
            self.baseline_false_negatives += int(not is_flagged_baseline)
            self.agentic_true_positives += int(is_flagged_agentic)
            self.agentic_false_negatives += int(not is_flagged_agentic)
        else:
            self.total_legitimate += 1
            self.baseline_false_positives += int(is_flagged_baseline)
            self.baseline_true_negatives += int(not is_flagged_baseline)
            self.agentic_false_positives += int(is_flagged_agentic)
            self.agentic_true_negatives += int(not is_flagged_agentic)

        result.agreement = result.baseline_action == result.agentic_action
        if result.agreement:
            self.agreement_count += 1

        # Agentic is "better" if it correctly approves a legit txn that baseline flagged
        if not result.is_fraud and is_flagged_baseline and not is_flagged_agentic:
            result.agentic_better = True
            self.agentic_improvements += 1

    def _safe_div(self, num: int, den: int) -> float:
        return round(num / den, 4) if den > 0 else 0.0

    def summary(self) -> Dict:
        return {
            "total_transactions": self.total_transactions,
            "total_fraud": self.total_fraud,
            "total_legitimate": self.total_legitimate,
            "baseline": {
                "precision": self._safe_div(
                    self.baseline_true_positives, self.baseline_true_positives + self.baseline_false_positives
                ),
                "recall": self._safe_div(
                    self.baseline_true_positives, self.baseline_true_positives + self.baseline_false_negatives
                ),
                "false_positive_rate": self._safe_div(self.baseline_false_positives, self.total_legitimate),
                "true_positives": self.baseline_true_positives,
                "false_positives": self.baseline_false_positives,
            },
            "agentic": {
                "precision": self._safe_div(
                    self.agentic_true_positives, self.agentic_true_positives + self.agentic_false_positives
                ),
                "recall": self._safe_div(
                    self.agentic_true_positives, self.agentic_true_positives + self.agentic_false_negatives
                ),
                "false_positive_rate": self._safe_div(self.agentic_false_positives, self.total_legitimate),
                "true_positives": self.agentic_true_positives,
                "false_positives": self.agentic_false_positives,
            },
            "agreement_rate": self._safe_div(self.agreement_count, self.total_transactions),
            "agentic_improvements": self.agentic_improvements,
            "improvement_details": [
                {"txn_id": r.txn_id, "reasoning": r.agentic_reasoning} for r in self.results if r.agentic_better
            ],
        }
