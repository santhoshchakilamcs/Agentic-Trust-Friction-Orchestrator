"""
Agentic Risk & Retention Orchestrator — Demo Runner

Runs synthetic transactions through both the legacy baseline model
and the full agentic pipeline, then compares results in shadow mode.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_orchestrator.compliance.pii_masker import PIIMasker
from agentic_orchestrator.data.generator import Transaction, generate_transactions
from agentic_orchestrator.evaluation.baseline_model import BaselineModel
from agentic_orchestrator.evaluation.shadow_mode import ShadowModeReport, ShadowResult
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.orchestrator.engine import OrchestratorEngine

console = Console()


def txn_to_state(txn: Transaction) -> TransactionState:
    """Convert a Transaction dataclass to a TransactionState."""
    return TransactionState(
        txn_id=txn.txn_id,
        user_id=txn.user_id,
        amount_usd=txn.amount_usd,
        recipient_name=txn.recipient_name,
        recipient_country=txn.recipient_country,
        corridor=txn.corridor,
        device_id=txn.device_id,
        ip_address=txn.ip_address,
        timestamp=txn.timestamp,
        is_new_recipient=txn.is_new_recipient,
    )


def display_transaction_result(txn: Transaction, baseline: dict, agentic_state: TransactionState, idx: int):
    """Display a single transaction's processing results."""
    color = "red" if txn.is_fraud else "green"
    label = f"🔴 FRAUD ({txn.label})" if txn.is_fraud else "🟢 LEGITIMATE"

    console.print(f"\n{'=' * 80}")
    console.print(f"[bold]Transaction #{idx + 1}[/bold] | {txn.txn_id} | {label}", style=color)
    console.print(f"  User: {txn.user_id} | ${txn.amount_usd:.2f} → {txn.recipient_name} ({txn.recipient_country})")
    console.print(f"  Corridor: {txn.corridor} | New recipient: {txn.is_new_recipient}")

    # Baseline result
    b_color = "green" if baseline["action"] == "APPROVE" else "yellow" if baseline["action"] == "CHALLENGE" else "red"
    console.print(
        f"  [dim]Baseline:[/dim]  [{b_color}]{baseline['action']}[/{b_color}] (score: {baseline['score']:.4f})"
    )

    # Agentic result
    a_color = (
        "green"
        if agentic_state.final_action == "APPROVE"
        else "yellow"
        if agentic_state.final_action == "CHALLENGE"
        else "red"
    )
    console.print(
        f"  [dim]Agentic:[/dim]   [{a_color}]{agentic_state.final_action}[/{a_color}] (score: {agentic_state.risk_score:.4f})"
    )

    if agentic_state.challenge_message:
        console.print(f"  [dim]Message:[/dim]  {agentic_state.challenge_message}")

    # Show processing log
    if agentic_state.processing_log:
        console.print("  [dim]Agent Log:[/dim]")
        for log_entry in agentic_state.processing_log[-5:]:  # Last 5 entries
            console.print(f"    {log_entry}")


def display_shadow_report(report: ShadowModeReport):
    """Display the shadow mode comparison report."""
    summary = report.summary()

    console.print("\n")
    console.print(Panel("[bold]SHADOW MODE EVALUATION REPORT[/bold]", style="cyan", box=box.DOUBLE))

    # Overview table
    overview = Table(title="Overview", box=box.ROUNDED)
    overview.add_column("Metric", style="bold")
    overview.add_column("Value", justify="right")
    overview.add_row("Total Transactions", str(summary["total_transactions"]))
    overview.add_row("Fraud Cases", str(summary["total_fraud"]))
    overview.add_row("Legitimate Cases", str(summary["total_legitimate"]))
    overview.add_row("Agreement Rate", f"{summary['agreement_rate']:.1%}")
    overview.add_row("Agentic Improvements", str(summary["agentic_improvements"]))
    console.print(overview)

    # Comparison table
    comparison = Table(title="Baseline vs Agentic Performance", box=box.ROUNDED)
    comparison.add_column("Metric", style="bold")
    comparison.add_column("Baseline", justify="right")
    comparison.add_column("Agentic", justify="right")
    comparison.add_column("Δ", justify="right")

    for metric in ["precision", "recall", "false_positive_rate"]:
        b_val = summary["baseline"][metric]
        a_val = summary["agentic"][metric]
        delta = a_val - b_val
        d_color = (
            "green"
            if (metric != "false_positive_rate" and delta > 0) or (metric == "false_positive_rate" and delta < 0)
            else "red"
            if delta != 0
            else "white"
        )
        comparison.add_row(
            metric.replace("_", " ").title(), f"{b_val:.4f}", f"{a_val:.4f}", f"[{d_color}]{delta:+.4f}[/{d_color}]"
        )

    comparison.add_row(
        "True Positives", str(summary["baseline"]["true_positives"]), str(summary["agentic"]["true_positives"]), ""
    )
    comparison.add_row(
        "False Positives", str(summary["baseline"]["false_positives"]), str(summary["agentic"]["false_positives"]), ""
    )
    console.print(comparison)

    # Improvement examples
    if summary["improvement_details"]:
        console.print(
            "\n[bold cyan]Agentic Improvements (Legit txns correctly approved that baseline would have flagged):[/bold cyan]"
        )
        for detail in summary["improvement_details"][:5]:
            console.print(f"  ✅ {detail['txn_id']}: {detail['reasoning'][:120]}...")

    # PII masking demo
    console.print("\n[bold cyan]PII Masking Demo:[/bold cyan]")
    sample = {
        "user_id": "USR-ABC123",
        "ip_address": "192.168.1.1",
        "recipient_name": "Elena Santos",
        "amount_usd": 500.0,
    }
    masked = PIIMasker.mask_dict(sample)
    console.print(f"  Original: {sample}")
    console.print(f"  Masked:   {masked}")


def display_judge_report(judge_results: list):
    """Display LLM-as-a-Judge evaluation summary."""
    if not judge_results:
        return
    console.print("\n")
    console.print(Panel("[bold]LLM-AS-A-JUDGE EVALUATION[/bold]", style="magenta", box=box.DOUBLE))

    grade_counts = {}
    total_score = 0.0
    issues_all = []
    for r in judge_results:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
        total_score += r["overall_score"]
        issues_all.extend(r.get("issues", []))

    avg_score = total_score / len(judge_results)

    tbl = Table(title="Judge Summary", box=box.ROUNDED)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Transactions Evaluated", str(len(judge_results)))
    tbl.add_row("Average Quality Score", f"{avg_score:.4f}")
    for grade in ["A", "B", "C", "D", "F"]:
        if grade in grade_counts:
            tbl.add_row(f"Grade {grade}", str(grade_counts[grade]))
    tbl.add_row("Total Issues Found", str(len(issues_all)))
    console.print(tbl)

    # Show top issues
    if issues_all:
        from collections import Counter

        top_issues = Counter(issues_all).most_common(5)
        console.print("\n[bold magenta]Top Issues Identified by Judge:[/bold magenta]")
        for issue, count in top_issues:
            console.print(f"  ⚠️  ({count}x) {issue}")


def display_hitl_report(review_log: list):
    """Display HITL review summary."""
    if not review_log:
        return
    console.print("\n")
    console.print(Panel("[bold]HUMAN-IN-THE-LOOP REVIEW LOG[/bold]", style="yellow", box=box.DOUBLE))
    tbl = Table(title="HITL Decisions", box=box.ROUNDED)
    tbl.add_column("Transaction", style="bold")
    tbl.add_column("Decision", justify="center")
    tbl.add_column("Original Risk", justify="right")
    tbl.add_column("Notes")
    for entry in review_log:
        d_color = "green" if entry["decision"] == "APPROVE" else "red" if entry["decision"] == "REJECT" else "yellow"
        tbl.add_row(
            entry["txn_id"],
            f"[{d_color}]{entry['decision']}[/{d_color}]",
            f"{entry['original_risk_action']} ({entry['original_risk_score']:.4f})",
            entry.get("notes", "")[:60],
        )
    console.print(tbl)


def main():
    console.print(
        Panel(
            "[bold white]🤖 AGENTIC RISK & RETENTION ORCHESTRATOR[/bold white]\n"
            "[dim]Multi-Agent System for Cross-Border Payment Fraud Detection[/dim]\n"
            "[dim]Shadow Mode: Comparing Baseline vs. Agentic Decisions[/dim]",
            style="blue",
            box=box.DOUBLE,
        )
    )

    # 1. Generate synthetic data
    console.print("\n[bold]Step 1:[/bold] Generating synthetic transaction data...")
    transactions, users = generate_transactions(n_total=50, fraud_ratio=0.15, n_users=10)
    console.print(f"  Generated {len(transactions)} transactions from {len(users)} users")

    # 2. Initialize systems
    console.print("[bold]Step 2:[/bold] Initializing memory and agents...")
    memory = LongTermMemory()
    memory.ingest_user_profiles(users)
    # HITL disabled for batch demo (set hitl_enabled=True for interactive mode)
    engine = OrchestratorEngine(memory, hitl_enabled=False, judge_enabled=True)
    baseline = BaselineModel()
    report = ShadowModeReport()
    judge_results = []
    console.print("  ✅ All systems initialized (HITL=batch-auto, LLM-Judge=enabled)")

    # 3. Process transactions
    console.print("[bold]Step 3:[/bold] Processing transactions through both pipelines...\n")

    for idx, txn in enumerate(transactions):
        # Baseline prediction
        baseline_result = baseline.predict(txn)

        # Agentic pipeline
        state = txn_to_state(txn)
        agentic_result = engine.process_transaction(state)

        # Display first 10 transactions in detail
        if idx < 10:
            display_transaction_result(txn, baseline_result, agentic_result, idx)

        # Collect LLM-Judge results
        if agentic_result.judge_score is not None:
            judge_results.append(
                {
                    "txn_id": txn.txn_id,
                    "overall_score": agentic_result.judge_score,
                    "grade": agentic_result.judge_reasoning_grade,
                    "feedback": agentic_result.judge_feedback,
                    "issues": [],  # issues are in the feedback string
                }
            )

        # Record in shadow mode
        report.add_result(
            ShadowResult(
                txn_id=txn.txn_id,
                is_fraud=txn.is_fraud,
                ground_truth_label=txn.label,
                baseline_action=baseline_result["action"],
                baseline_score=baseline_result["score"],
                agentic_action=agentic_result.final_action,
                agentic_score=agentic_result.risk_score,
                agentic_reasoning=agentic_result.final_reasoning,
            )
        )

    # 4. Display shadow mode report
    console.print(f"\n[dim](Showing first 10 of {len(transactions)} transactions in detail)[/dim]")
    display_shadow_report(report)

    # 5. Display LLM-as-a-Judge report
    display_judge_report(judge_results)

    # 6. Display HITL review log
    display_hitl_report(engine.hitl.review_log)

    # Cleanup
    memory.clear()
    console.print("\n[bold green]✅ Demo complete![/bold green]")


if __name__ == "__main__":
    main()
