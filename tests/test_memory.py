"""Tests for short-term and long-term memory."""

import pytest

from agentic_orchestrator.data.generator import UserProfile
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.memory.short_term import TransactionState


class TestTransactionState:
    def test_default_values(self):
        state = TransactionState()
        assert state.txn_id == ""
        assert state.amount_usd == 0.0
        assert state.investigator_score == 0.0
        assert state.risk_score == 0.0
        assert state.final_action == ""
        assert state.processing_log == []
        assert state.compliance_override is None

    def test_log_appends_formatted_entry(self):
        state = TransactionState()
        state.log("TestAgent", "Something happened")
        assert len(state.processing_log) == 1
        assert "[TestAgent] Something happened" in state.processing_log[0]

    def test_to_dict_returns_all_fields(self):
        state = TransactionState(txn_id="TXN-123", amount_usd=500.0)
        d = state.to_dict()
        assert d["txn_id"] == "TXN-123"
        assert d["amount_usd"] == 500.0
        assert "investigator_score" in d
        assert "risk_action" in d

    def test_state_is_mutable(self):
        state = TransactionState()
        state.risk_score = 0.75
        state.final_action = "BLOCK"
        assert state.risk_score == 0.75
        assert state.final_action == "BLOCK"


class TestLongTermMemory:
    @pytest.fixture
    def memory(self):
        mem = LongTermMemory()
        yield mem
        mem.clear()

    @pytest.fixture
    def sample_profiles(self):
        return [
            UserProfile(
                user_id="USR-TEST001",
                name="Maria Santos",
                country="US",
                typical_amount=300.0,
                typical_recipients=["Elena Santos", "Pedro Garcia"],
                typical_corridor="US-PH",
                registered_device_id="DEV-ABC123",
                registered_ip_prefix="192.168.1",
                life_events=[{"event": "Sending money for sister's wedding", "date": "2026-01-01"}],
            ),
            UserProfile(
                user_id="USR-TEST002",
                name="Raj Patel",
                country="GB",
                typical_amount=500.0,
                typical_recipients=["Vikram Patel"],
                typical_corridor="GB-IN",
                registered_device_id="DEV-DEF456",
                registered_ip_prefix="10.0.0",
                life_events=[],
            ),
        ]

    def test_ingest_and_retrieve_profile(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        profile = memory.get_user_profile("USR-TEST001")
        assert profile is not None
        assert profile["metadata"]["user_id"] == "USR-TEST001"
        assert "Maria Santos" in profile["document"]

    def test_retrieve_nonexistent_user(self, memory):
        profile = memory.get_user_profile("USR-NONEXISTENT")
        assert profile is None

    def test_query_user_context(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        result = memory.query_user_context("USR-TEST001", "transfer to Philippines")
        assert result["found"] is True
        assert len(result["documents"]) > 0

    def test_query_nonexistent_user_context(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        result = memory.query_user_context("USR-NONEXISTENT", "anything")
        assert result["found"] is False

    def test_life_events_in_document(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        profile = memory.get_user_profile("USR-TEST001")
        assert "wedding" in profile["document"].lower()

    def test_no_life_events_noted(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        profile = memory.get_user_profile("USR-TEST002")
        assert "No known life events" in profile["document"]

    def test_add_feedback(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        memory.add_feedback("USR-TEST001", "This user regularly sends large amounts for family")
        profile = memory.get_user_profile("USR-TEST001")
        assert "Feedback" in profile["document"]

    def test_clear_resets_memory(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        memory.clear()
        profile = memory.get_user_profile("USR-TEST001")
        assert profile is None

    def test_upsert_updates_existing(self, memory, sample_profiles):
        memory.ingest_user_profiles(sample_profiles)
        # Modify and re-ingest
        sample_profiles[0].typical_amount = 999.0
        memory.ingest_user_profiles(sample_profiles)
        profile = memory.get_user_profile("USR-TEST001")
        assert "$999.00" in profile["document"]
