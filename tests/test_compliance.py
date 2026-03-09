"""Tests for the compliance firewall and PII masker."""

from agentic_orchestrator.compliance.firewall import ComplianceFirewall
from agentic_orchestrator.compliance.pii_masker import PIIMasker
from agentic_orchestrator.config import (
    ACTION_APPROVE,
    ACTION_BLOCK,
    ACTION_ESCALATE,
    AML_THRESHOLD_USD,
)
from agentic_orchestrator.memory.short_term import TransactionState


def _make_state(**overrides) -> TransactionState:
    defaults = dict(
        txn_id="TXN-COMP01",
        user_id="USR-COMP01",
        amount_usd=200.0,
        recipient_name="Elena Santos",
        recipient_country="PH",
        corridor="US-PH",
        device_id="DEV-ABC",
        ip_address="10.0.0.1",
        timestamp="2026-03-09T10:00:00",
        is_new_recipient=False,
    )
    defaults.update(overrides)
    return TransactionState(**defaults)


class TestComplianceFirewall:
    def test_normal_transaction_passes(self):
        fw = ComplianceFirewall()
        state = _make_state()
        result = fw.check(state)
        assert result.compliance_override is None
        assert len(result.compliance_flags) == 0

    def test_sanctioned_country_blocks(self):
        fw = ComplianceFirewall()
        for country in ["KP", "IR", "SY", "CU"]:
            state = _make_state(recipient_country=country)
            result = fw.check(state)
            assert result.compliance_override == ACTION_BLOCK
            assert any("Sanctioned" in f for f in result.compliance_flags)

    def test_aml_threshold_escalates(self):
        fw = ComplianceFirewall()
        state = _make_state(amount_usd=AML_THRESHOLD_USD + 1)
        result = fw.check(state)
        assert result.compliance_override == ACTION_ESCALATE
        assert any("AML" in f for f in result.compliance_flags)

    def test_just_below_aml_no_override(self):
        fw = ComplianceFirewall()
        state = _make_state(amount_usd=AML_THRESHOLD_USD - 1)
        result = fw.check(state)
        assert result.compliance_override is None

    def test_structuring_detection(self):
        fw = ComplianceFirewall()
        # 90% of AML threshold + new recipient
        state = _make_state(
            amount_usd=AML_THRESHOLD_USD * 0.95,
            is_new_recipient=True,
        )
        result = fw.check(state)
        assert any("structuring" in f.lower() for f in result.compliance_flags)

    def test_apply_override_blocks(self):
        fw = ComplianceFirewall()
        state = _make_state(recipient_country="KP")
        state = fw.check(state)
        state.risk_action = ACTION_APPROVE  # Agent says approve
        state = fw.apply_override(state)
        assert state.final_action == ACTION_BLOCK  # Compliance overrides
        assert "COMPLIANCE OVERRIDE" in state.final_reasoning

    def test_apply_override_no_override(self):
        fw = ComplianceFirewall()
        state = _make_state()
        state = fw.check(state)
        state.risk_action = ACTION_APPROVE
        state = fw.apply_override(state)
        assert state.final_action == ""  # No override applied


class TestPIIMasker:
    def test_mask_ip_address(self):
        result = PIIMasker.mask_string("User IP is 192.168.1.100")
        assert "192.168.1.100" not in result
        assert result.startswith("User IP is ip_address_")

    def test_mask_email(self):
        result = PIIMasker.mask_string("Contact user@example.com for help")
        assert "user@example.com" not in result
        assert "email_" in result

    def test_mask_card_number(self):
        result = PIIMasker.mask_string("Card: 4111-1111-1111-1111")
        assert "4111" not in result
        assert "card_number_" in result

    def test_mask_dict_sensitive_fields(self):
        data = {
            "user_id": "USR-123",
            "ip_address": "10.0.0.1",
            "recipient_name": "Elena Santos",
            "amount_usd": 500.0,
        }
        masked = PIIMasker.mask_dict(data)
        assert masked["ip_address"].startswith("ip_address_")
        assert masked["recipient_name"].startswith("recipient_name_")
        assert masked["user_id"] == "USR-123"  # Not in sensitive fields
        assert masked["amount_usd"] == 500.0

    def test_mask_nested_dict(self):
        data = {"outer": {"ip_address": "1.2.3.4", "name": "Test"}}
        masked = PIIMasker.mask_dict(data)
        assert masked["outer"]["ip_address"].startswith("ip_address_")

    def test_hash_deterministic(self):
        """Same input always produces the same hash."""
        h1 = PIIMasker._hash("192.168.1.1", prefix="ip")
        h2 = PIIMasker._hash("192.168.1.1", prefix="ip")
        assert h1 == h2

    def test_hash_different_inputs_differ(self):
        """Different inputs produce different hashes."""
        h1 = PIIMasker._hash("192.168.1.1", prefix="ip")
        h2 = PIIMasker._hash("10.0.0.1", prefix="ip")
        assert h1 != h2

    def test_mask_for_logging(self):
        state_dict = {
            "txn_id": "TXN-123",
            "ip_address": "192.168.1.1",
            "device_id": "DEV-ABC",
            "recipient_name": "Maria Santos",
        }
        masked = PIIMasker.mask_for_logging(state_dict)
        assert "192.168.1.1" not in str(masked)
        assert "Maria Santos" not in str(masked)


class TestHITLReviewer:
    """Tests for the Human-in-the-Loop reviewer."""

    def test_needs_review_on_escalation(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(enabled=False)
        state = _make_state()
        state.compliance_override = ACTION_ESCALATE
        assert reviewer.needs_review(state) is True

    def test_no_review_for_normal(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(enabled=False)
        state = _make_state()
        assert reviewer.needs_review(state) is False

    def test_disabled_auto_approves(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(enabled=False)
        state = _make_state(amount_usd=5000.0)
        state.compliance_override = ACTION_ESCALATE
        state.risk_action = ACTION_ESCALATE
        state.risk_score = 0.8
        result = reviewer.review(state)
        assert result.hitl_decision == "APPROVE"
        assert result.final_action == ACTION_APPROVE
        assert "AUTO-APPROVED" in result.hitl_reviewer_notes

    def test_callback_approve(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(review_callback=lambda s: "APPROVE", enabled=True)
        state = _make_state(amount_usd=5000.0)
        state.compliance_override = ACTION_ESCALATE
        state.risk_action = ACTION_ESCALATE
        result = reviewer.review(state)
        assert result.hitl_decision == "APPROVE"
        assert result.final_action == ACTION_APPROVE

    def test_callback_reject(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(review_callback=lambda s: "REJECT", enabled=True)
        state = _make_state(amount_usd=5000.0)
        state.compliance_override = ACTION_ESCALATE
        state.risk_action = ACTION_ESCALATE
        result = reviewer.review(state)
        assert result.hitl_decision == "REJECT"
        assert result.final_action == ACTION_BLOCK

    def test_review_log_recorded(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(review_callback=lambda s: "APPROVE", enabled=True)
        state = _make_state(txn_id="TXN-HITL-01")
        state.compliance_override = ACTION_ESCALATE
        state.risk_action = ACTION_ESCALATE
        state.risk_score = 0.7
        reviewer.review(state)
        assert len(reviewer.review_log) == 1
        assert reviewer.review_log[0]["txn_id"] == "TXN-HITL-01"
        assert reviewer.review_log[0]["decision"] == "APPROVE"

    def test_skip_if_not_needed(self):
        from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer

        reviewer = HITLReviewer(review_callback=lambda s: "REJECT", enabled=True)
        state = _make_state()
        result = reviewer.review(state)
        assert result.hitl_decision is None
        assert result.final_action == ""  # unchanged
