"""Single-transaction trace export for the Agentic Trust & Friction Orchestrator.

This module is intentionally standalone: it does not call any LLM and
does not change the agent pipeline. It takes the decisions a run
already produced (agent name, decision, risk contribution, note) and
writes them as one JSON trace file for a single test transaction.

All data in the example trace is synthetic. Nothing here is a real
payment, a real user, or a real risk decision. PII in real runs must
already have been hashed by the pipeline before reaching this module;
do not pass raw IP addresses, device IDs, or recipient names here.
"""

import json
from pathlib import Path
from typing import Any


def build_trace(
    txn_id: str,
    agent_decisions: list[dict[str, Any]],
    final_action: str,
    risk_score: float,
    synthetic: bool = True,
) -> dict[str, Any]:
    """Build one trace dict from already-produced agent decisions.

    Each decision entry should have at least ``agent`` and ``decision``;
    ``risk_contribution`` and ``note`` are optional.
    """
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError("risk_score must be between 0.0 and 1.0")
    decisions = []
    for d in agent_decisions:
        if "agent" not in d or "decision" not in d:
            raise ValueError("each agent decision needs 'agent' and 'decision'")
        decisions.append(
            {
                "agent": d["agent"],
                "decision": d["decision"],
                "risk_contribution": d.get("risk_contribution"),
                "note": d.get("note", ""),
            }
        )
    return {
        "txn_id": txn_id,
        "synthetic_data_only": synthetic,
        "final_action": final_action,
        "risk_score": risk_score,
        "agent_decisions": decisions,
    }


def export_trace(trace: dict[str, Any], path: str | Path) -> Path:
    """Write a trace dict to ``path`` as JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return out


def example_trace() -> dict[str, Any]:
    """A synthetic example trace, matching the README's friction tiers."""
    return build_trace(
        txn_id="SYNTH-TEST-001",
        agent_decisions=[
            {"agent": "Compliance Firewall", "decision": "PASS",
             "note": "Not a sanctioned country and below the AML threshold."},
            {"agent": "Investigator", "decision": "AMOUNT_ANOMALY_LOW",
             "risk_contribution": 0.10, "note": "Amount within the user's usual range."},
            {"agent": "Context", "decision": "KNOWN_RECIPIENT",
             "risk_contribution": -0.05, "note": "Recipient seen in past history (synthetic)."},
            {"agent": "Risk Scorer", "decision": "CHALLENGE",
             "risk_contribution": 0.45, "note": "Score 0.45 falls in the 0.3-0.6 CHALLENGE band."},
            {"agent": "Communication", "decision": "SEND_2FA_MESSAGE",
             "note": "Friendly verification message for the challenge step."},
        ],
        final_action="CHALLENGE",
        risk_score=0.45,
    )


if __name__ == "__main__":
    print(json.dumps(example_trace(), indent=2))
