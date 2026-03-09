"""
Baseline Model: A simple static fraud detection model (the 'Seed 42' legacy approach).
This represents what Remitly might currently use — a context-blind, threshold-based model.
Used as the comparison baseline for shadow mode evaluation.
"""

from agentic_orchestrator.data.generator import Transaction
from agentic_orchestrator.config import (
    AML_THRESHOLD_USD,
    SANCTIONS_COUNTRIES,
    ACTION_APPROVE,
    ACTION_CHALLENGE,
    ACTION_ESCALATE,
    ACTION_BLOCK,
)


class BaselineModel:
    """
    Static, rule-based fraud model. No context awareness, no memory.
    This is the 'before' system we're trying to beat.
    """

    def predict(self, txn: Transaction) -> dict:
        """Return a decision dict with action, score, and reasoning."""
        score = 0.0
        flags = []

        # Rule 1: Sanctions
        if txn.recipient_country in SANCTIONS_COUNTRIES:
            return {
                "action": ACTION_BLOCK,
                "score": 1.0,
                "flags": [f"Sanctioned country: {txn.recipient_country}"],
                "reasoning": "Blocked due to sanctioned destination.",
            }

        # Rule 2: AML threshold
        if txn.amount_usd >= AML_THRESHOLD_USD:
            score += 0.4
            flags.append(f"Amount ${txn.amount_usd:.2f} exceeds AML threshold")

        # Rule 3: High amount (static threshold)
        if txn.amount_usd > 1500:
            score += 0.2
            flags.append(f"High amount: ${txn.amount_usd:.2f}")

        # Rule 4: New recipient
        if txn.is_new_recipient:
            score += 0.25
            flags.append("New recipient")

        # Rule 5: High amount + new recipient combo
        if txn.amount_usd > 1000 and txn.is_new_recipient:
            score += 0.15
            flags.append("High amount to new recipient")

        # Clamp
        score = min(1.0, score)

        # Decision
        if score < 0.3:
            action = ACTION_APPROVE
        elif score < 0.6:
            action = ACTION_CHALLENGE
        else:
            action = ACTION_ESCALATE

        return {
            "action": action,
            "score": round(score, 4),
            "flags": flags,
            "reasoning": f"Static model score: {score:.4f}. Flags: {'; '.join(flags) if flags else 'None'}.",
        }

