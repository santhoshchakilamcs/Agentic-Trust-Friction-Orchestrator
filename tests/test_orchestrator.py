"""Tests for the LangGraph orchestrator and shadow mode evaluation (heuristic mode)."""

import pytest
from unittest.mock import patch
from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.data.generator import UserProfile, Transaction
from agentic_orchestrator.orchestrator.engine import OrchestratorEngine
from agentic_orchestrator.evaluation.baseline_model import BaselineModel
from agentic_orchestrator.evaluation.shadow_mode import ShadowModeReport, ShadowResult
from agentic_orchestrator.config import (
    ACTION_APPROVE, ACTION_BLOCK, ACTION_CHALLENGE, ACTION_ESCALATE,
    AML_THRESHOLD_USD,
)


# Force all agents to use heuristic fallback (no real API calls in unit tests)
@pytest.fixture(autouse=True)
def _disable_llm():
    with patch("agentic_orchestrator.agents.investigator.is_available", return_value=False), \
         patch("agentic_orchestrator.agents.context.is_available", return_value=False), \
         patch("agentic_orchestrator.agents.risk_scorer.is_available", return_value=False), \
         patch("agentic_orchestrator.agents.communication.is_available", return_value=False), \
         patch("agentic_orchestrator.evaluation.llm_judge.is_available", return_value=False):
        yield


@pytest.fixture
def engine():
    mem = LongTermMemory()
    mem.ingest_user_profiles([
        UserProfile(
            user_id="USR-ORCH01", name="Maria Santos", country="US",
            typical_amount=300.0, typical_recipients=["Elena Santos"],
            typical_corridor="US-PH", registered_device_id="DEV-REG",
            registered_ip_prefix="192.168.1",
            life_events=[{"event": "Monthly support for parents", "date": "2026-01-01"}],
        ),
    ])
    eng = OrchestratorEngine(mem)
    yield eng
    mem.clear()


def _make_state(**overrides) -> TransactionState:
    defaults = dict(
        txn_id="TXN-ORCH01", user_id="USR-ORCH01", amount_usd=250.0,
        recipient_name="Elena Santos", recipient_country="PH",
        corridor="US-PH", device_id="DEV-REG", ip_address="192.168.1.50",
        timestamp="2026-03-09T10:00:00", is_new_recipient=False,
    )
    defaults.update(overrides)
    return TransactionState(**defaults)


class TestOrchestratorEngine:
    def test_normal_transaction_approved(self, engine):
        state = _make_state()
        result = engine.process_transaction(state)
        assert result.final_action == ACTION_APPROVE

    def test_sanctioned_country_blocked(self, engine):
        state = _make_state(recipient_country="KP", corridor="US-KP")
        result = engine.process_transaction(state)
        assert result.final_action == ACTION_BLOCK

    def test_aml_threshold_escalated(self, engine):
        """AML threshold triggers HITL review. With HITL disabled, it auto-approves."""
        state = _make_state(amount_usd=AML_THRESHOLD_USD + 500)
        result = engine.process_transaction(state)
        # HITL is disabled in default engine fixture → auto-approved
        assert result.hitl_required is True
        assert result.hitl_decision == "APPROVE"
        assert result.final_action == ACTION_APPROVE
        assert "HUMAN APPROVED" in result.final_reasoning

    def test_processing_log_populated(self, engine):
        state = _make_state()
        result = engine.process_transaction(state)
        assert len(result.processing_log) > 0
        assert any("Orchestrator" in log for log in result.processing_log)

    def test_final_reasoning_populated(self, engine):
        state = _make_state()
        result = engine.process_transaction(state)
        assert result.final_reasoning != ""

    def test_compliance_overrides_agent(self, engine):
        """Even if agents would approve, compliance should override."""
        state = _make_state(recipient_country="IR", corridor="US-IR")
        result = engine.process_transaction(state)
        assert result.final_action == ACTION_BLOCK
        assert "COMPLIANCE OVERRIDE" in result.final_reasoning


class TestBaselineModel:
    def test_approve_normal_transaction(self):
        model = BaselineModel()
        txn = Transaction(
            txn_id="TXN-B01", user_id="USR-01", amount_usd=200.0,
            recipient_name="Elena", recipient_country="PH", corridor="US-PH",
            device_id="DEV-1", ip_address="1.2.3.4",
            timestamp="2026-03-09T10:00:00", is_new_recipient=False,
        )
        result = model.predict(txn)
        assert result["action"] == ACTION_APPROVE
        assert result["score"] < 0.3

    def test_block_sanctioned_country(self):
        model = BaselineModel()
        txn = Transaction(
            txn_id="TXN-B02", user_id="USR-01", amount_usd=200.0,
            recipient_name="Elena", recipient_country="KP", corridor="US-KP",
            device_id="DEV-1", ip_address="1.2.3.4",
            timestamp="2026-03-09T10:00:00", is_new_recipient=False,
        )
        result = model.predict(txn)
        assert result["action"] == ACTION_BLOCK

    def test_high_amount_new_recipient_escalates(self):
        model = BaselineModel()
        txn = Transaction(
            txn_id="TXN-B03", user_id="USR-01", amount_usd=3500.0,
            recipient_name="Unknown", recipient_country="PH", corridor="US-PH",
            device_id="DEV-1", ip_address="1.2.3.4",
            timestamp="2026-03-09T10:00:00", is_new_recipient=True,
        )
        result = model.predict(txn)
        assert result["action"] in (ACTION_CHALLENGE, ACTION_ESCALATE)


class TestShadowModeReport:
    def test_correct_classification_counts(self):
        report = ShadowModeReport()
        # Fraud correctly caught by both
        report.add_result(ShadowResult(
            txn_id="T1", is_fraud=True, ground_truth_label="fraud",
            baseline_action=ACTION_ESCALATE, baseline_score=0.8,
            agentic_action=ACTION_ESCALATE, agentic_score=0.7,
            agentic_reasoning="High risk",
        ))
        # Legit correctly approved by both
        report.add_result(ShadowResult(
            txn_id="T2", is_fraud=False, ground_truth_label="legit",
            baseline_action=ACTION_APPROVE, baseline_score=0.1,
            agentic_action=ACTION_APPROVE, agentic_score=0.05,
            agentic_reasoning="Low risk",
        ))
        # Legit flagged by baseline but approved by agentic (improvement)
        report.add_result(ShadowResult(
            txn_id="T3", is_fraud=False, ground_truth_label="legit",
            baseline_action=ACTION_CHALLENGE, baseline_score=0.5,
            agentic_action=ACTION_APPROVE, agentic_score=0.2,
            agentic_reasoning="Context showed known recipient",
        ))

        summary = report.summary()
        assert summary["total_transactions"] == 3
        assert summary["agentic_improvements"] == 1
        assert summary["baseline"]["false_positives"] == 1
        assert summary["agentic"]["false_positives"] == 0
        assert summary["baseline"]["true_positives"] == 1
        assert summary["agentic"]["true_positives"] == 1

    def test_agreement_rate(self):
        report = ShadowModeReport()
        report.add_result(ShadowResult(
            txn_id="T1", is_fraud=False, ground_truth_label="legit",
            baseline_action=ACTION_APPROVE, baseline_score=0.1,
            agentic_action=ACTION_APPROVE, agentic_score=0.1,
            agentic_reasoning="",
        ))
        report.add_result(ShadowResult(
            txn_id="T2", is_fraud=False, ground_truth_label="legit",
            baseline_action=ACTION_CHALLENGE, baseline_score=0.5,
            agentic_action=ACTION_APPROVE, agentic_score=0.1,
            agentic_reasoning="",
        ))
        summary = report.summary()
        assert summary["agreement_rate"] == 0.5


class TestLLMJudge:
    """Tests for the LLM-as-a-Judge evaluator."""

    def test_good_transaction_gets_high_score(self, engine):
        """A well-processed normal transaction should get grade A or B."""
        state = _make_state()
        result = engine.process_transaction(state)
        assert result.judge_score is not None
        assert result.judge_score >= 0.7
        assert result.judge_reasoning_grade in ("A", "B")

    def test_sanctioned_block_gets_good_score(self, engine):
        """Correctly blocking a sanctioned country should score well."""
        state = _make_state(recipient_country="KP", corridor="US-KP")
        result = engine.process_transaction(state)
        assert result.judge_score is not None
        assert result.judge_score >= 0.5

    def test_judge_populates_feedback(self, engine):
        state = _make_state()
        result = engine.process_transaction(state)
        assert result.judge_feedback != ""

    def test_judge_logs_entry(self, engine):
        state = _make_state()
        result = engine.process_transaction(state)
        assert any("LLM-Judge" in log for log in result.processing_log)

    def test_judge_disabled(self):
        """When judge_enabled=False, no judge evaluation occurs."""
        mem = LongTermMemory()
        mem.ingest_user_profiles([
            UserProfile(
                user_id="USR-NOJUDGE", name="Test", country="US",
                typical_amount=300.0, typical_recipients=["Recipient"],
                typical_corridor="US-PH", registered_device_id="DEV-1",
                registered_ip_prefix="10.0.0", life_events=[],
            ),
        ])
        eng = OrchestratorEngine(mem, judge_enabled=False)
        state = _make_state(user_id="USR-NOJUDGE")
        result = eng.process_transaction(state)
        assert result.judge_score is None
        mem.clear()

    def test_hitl_transaction_judged(self, engine):
        """AML transaction that goes through HITL should still be judged."""
        state = _make_state(amount_usd=AML_THRESHOLD_USD + 1000)
        result = engine.process_transaction(state)
        assert result.hitl_required is True
        assert result.judge_score is not None

    def test_judge_standalone_evaluation(self):
        """Test LLMJudge directly without the orchestrator."""
        from agentic_orchestrator.evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        state = _make_state()
        state.risk_reasoning = "Normal transaction to known recipient"
        state.risk_score = 0.15
        state.risk_action = ACTION_APPROVE
        state.investigator_flags = []
        state.context_findings = ["Known recipient", "Consistent corridor"]
        state.processing_log = [
            "[ComplianceFirewall] checks passed",
            "[InvestigatorAgent] low risk",
            "[ContextAgent] known recipient found",
            "[RiskScorer] approve",
        ]
        verdict = judge.evaluate(state)
        assert 0.0 <= verdict.overall_score <= 1.0
        assert verdict.grade in ("A", "B", "C", "D", "F")
        assert verdict.feedback != ""

