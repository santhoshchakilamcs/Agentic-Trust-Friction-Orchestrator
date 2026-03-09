"""Tests for all specialized agents (heuristic mode — no API calls)."""

from unittest.mock import patch

import pytest

from agentic_orchestrator.agents.communication import CommunicationAgent
from agentic_orchestrator.agents.context import ContextAgent
from agentic_orchestrator.agents.investigator import InvestigatorAgent
from agentic_orchestrator.agents.risk_scorer import RiskScorerAgent
from agentic_orchestrator.config import ACTION_APPROVE, ACTION_CHALLENGE, ACTION_ESCALATE
from agentic_orchestrator.data.generator import UserProfile
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.memory.short_term import TransactionState


# Force all agents to use heuristic fallback (no real API calls in unit tests)
@pytest.fixture(autouse=True)
def _disable_llm():
    with (
        patch("agentic_orchestrator.agents.investigator.is_available", return_value=False),
        patch("agentic_orchestrator.agents.context.is_available", return_value=False),
        patch("agentic_orchestrator.agents.risk_scorer.is_available", return_value=False),
        patch("agentic_orchestrator.agents.communication.is_available", return_value=False),
    ):
        yield


@pytest.fixture
def memory():
    mem = LongTermMemory()
    mem.ingest_user_profiles(
        [
            UserProfile(
                user_id="USR-AGENT01",
                name="Maria Santos",
                country="US",
                typical_amount=300.0,
                typical_recipients=["Elena Santos"],
                typical_corridor="US-PH",
                registered_device_id="DEV-KNOWN",
                registered_ip_prefix="192.168.1",
                life_events=[{"event": "Monthly support for parents", "date": "2026-01-01"}],
            ),
        ]
    )
    yield mem
    mem.clear()


def _make_state(**overrides) -> TransactionState:
    defaults = dict(
        txn_id="TXN-TEST001",
        user_id="USR-AGENT01",
        amount_usd=250.0,
        recipient_name="Elena Santos",
        recipient_country="PH",
        corridor="US-PH",
        device_id="DEV-KNOWN",
        ip_address="192.168.1.50",
        timestamp="2026-03-09T10:00:00",
        is_new_recipient=False,
    )
    defaults.update(overrides)
    return TransactionState(**defaults)


class TestInvestigatorAgent:
    def test_low_risk_normal_transaction(self, memory):
        agent = InvestigatorAgent(memory)
        state = _make_state()
        result = agent.run(state)
        assert result.investigator_score < 0.3
        assert len(result.investigator_flags) == 0 or all("trust" not in f for f in result.investigator_flags)

    def test_high_risk_amount_anomaly(self, memory):
        agent = InvestigatorAgent(memory)
        state = _make_state(amount_usd=2500.0)
        result = agent.run(state)
        assert result.investigator_score > 0.1
        assert any("Amount" in f for f in result.investigator_flags)

    def test_high_risk_new_device(self, memory):
        agent = InvestigatorAgent(memory)
        state = _make_state(device_id="DEV-UNKNOWN")
        result = agent.run(state)
        assert any("device" in f.lower() for f in result.investigator_flags)

    def test_sanctions_country_flag(self, memory):
        agent = InvestigatorAgent(memory)
        state = _make_state(recipient_country="KP")
        result = agent.run(state)
        assert any("Sanctioned" in f for f in result.investigator_flags)

    def test_new_recipient_flag(self, memory):
        agent = InvestigatorAgent(memory)
        state = _make_state(is_new_recipient=True, recipient_name="Unknown Person")
        result = agent.run(state)
        assert any("New recipient" in f for f in result.investigator_flags)

    def test_score_bounded_0_to_1(self, memory):
        agent = InvestigatorAgent(memory)
        # Extreme case: everything suspicious
        state = _make_state(
            amount_usd=50000,
            device_id="DEV-BAD",
            recipient_country="IR",
            is_new_recipient=True,
            timestamp="2026-03-09T03:00:00",
        )
        result = agent.run(state)
        assert 0.0 <= result.investigator_score <= 1.0


class TestContextAgent:
    def test_known_recipient_reduces_score(self, memory):
        agent = ContextAgent(memory)
        state = _make_state(recipient_name="Elena Santos", is_new_recipient=False)
        result = agent.run(state)
        assert result.context_score_adjustment < 0
        assert any("Known recipient" in f for f in result.context_findings)

    def test_consistent_corridor(self, memory):
        agent = ContextAgent(memory)
        state = _make_state()
        result = agent.run(state)
        assert any("Consistent corridor" in f for f in result.context_findings)

    def test_life_event_context(self, memory):
        agent = ContextAgent(memory)
        state = _make_state()
        result = agent.run(state)
        # User has "Monthly support for parents" life event
        if result.context_life_events:
            assert result.context_score_adjustment < 0

    def test_adjustment_bounded(self, memory):
        agent = ContextAgent(memory)
        state = _make_state()
        result = agent.run(state)
        assert -0.3 <= result.context_score_adjustment <= 0.3


class TestRiskScorerAgent:
    def test_approve_low_score(self):
        agent = RiskScorerAgent()
        state = _make_state()
        state.investigator_score = 0.1
        state.context_score_adjustment = -0.05
        result = agent.run(state)
        assert result.risk_action == ACTION_APPROVE
        assert result.risk_score < 0.3

    def test_challenge_medium_score(self):
        agent = RiskScorerAgent()
        state = _make_state()
        state.investigator_score = 0.4
        state.context_score_adjustment = 0.0
        result = agent.run(state)
        assert result.risk_action == ACTION_CHALLENGE

    def test_escalate_high_score(self):
        agent = RiskScorerAgent()
        state = _make_state()
        state.investigator_score = 0.7
        state.context_score_adjustment = 0.0
        result = agent.run(state)
        assert result.risk_action == ACTION_ESCALATE

    def test_context_can_lower_to_approve(self):
        agent = RiskScorerAgent()
        state = _make_state()
        state.investigator_score = 0.35
        state.context_score_adjustment = -0.15
        result = agent.run(state)
        assert result.risk_action == ACTION_APPROVE

    def test_reasoning_populated(self):
        agent = RiskScorerAgent()
        state = _make_state()
        state.investigator_score = 0.5
        state.investigator_flags = ["New recipient"]
        result = agent.run(state)
        assert result.risk_reasoning
        assert "New recipient" in result.risk_reasoning


class TestCommunicationAgent:
    def test_no_message_for_approve(self):
        agent = CommunicationAgent()
        state = _make_state()
        state.risk_action = ACTION_APPROVE
        result = agent.run(state)
        assert result.challenge_message == ""

    def test_challenge_generates_friendly_message(self):
        agent = CommunicationAgent()
        state = _make_state(amount_usd=500.0, recipient_name="Elena Santos")
        state.risk_action = ACTION_CHALLENGE
        result = agent.run(state)
        assert "Hi" in result.challenge_message
        assert "$500.00" in result.challenge_message
        assert "Elena Santos" in result.challenge_message

    def test_escalation_message(self):
        agent = CommunicationAgent()
        state = _make_state()
        state.risk_action = ACTION_ESCALATE
        state.compliance_flags = ["AML threshold exceeded"]
        result = agent.run(state)
        assert "review" in result.challenge_message.lower()

    def test_new_recipient_mentioned_in_challenge(self):
        agent = CommunicationAgent()
        state = _make_state(is_new_recipient=True, recipient_name="New Person")
        state.risk_action = ACTION_CHALLENGE
        result = agent.run(state)
        assert "first transfer" in result.challenge_message.lower()
