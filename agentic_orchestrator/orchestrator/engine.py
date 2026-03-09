"""
Orchestrator Engine: LangGraph-based workflow that coordinates all agents.
Implements the state machine for transaction decisioning.

Flow:
  compliance_check → investigator → context_agent → risk_scorer → communication → finalize
  (with compliance override at any point)
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from agentic_orchestrator.memory.short_term import TransactionState
from agentic_orchestrator.memory.long_term import LongTermMemory
from agentic_orchestrator.agents.investigator import InvestigatorAgent
from agentic_orchestrator.agents.context import ContextAgent
from agentic_orchestrator.agents.risk_scorer import RiskScorerAgent
from agentic_orchestrator.agents.communication import CommunicationAgent
from agentic_orchestrator.compliance.firewall import ComplianceFirewall
from agentic_orchestrator.compliance.hitl_reviewer import HITLReviewer, ReviewCallback
from agentic_orchestrator.evaluation.llm_judge import LLMJudge
from agentic_orchestrator.config import ACTION_APPROVE, ACTION_ESCALATE


class OrchestratorEngine:
    """
    The 'Brain' — manages the agentic workflow as a LangGraph state machine.
    """

    def __init__(
        self,
        memory: LongTermMemory,
        hitl_callback: ReviewCallback | None = None,
        hitl_enabled: bool = False,
        judge_enabled: bool = True,
    ):
        self.memory = memory
        self.investigator = InvestigatorAgent(memory)
        self.context_agent = ContextAgent(memory)
        self.risk_scorer = RiskScorerAgent()
        self.communication = CommunicationAgent()
        self.firewall = ComplianceFirewall()
        self.hitl = HITLReviewer(review_callback=hitl_callback, enabled=hitl_enabled)
        self.judge = LLMJudge() if judge_enabled else None
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # We use a dict-based state for LangGraph compatibility
        builder = StateGraph(dict)

        # --- Define nodes ---
        builder.add_node("compliance_check", self._node_compliance)
        builder.add_node("investigate", self._node_investigate)
        builder.add_node("context_lookup", self._node_context)
        builder.add_node("score_risk", self._node_risk_score)
        builder.add_node("communicate", self._node_communicate)
        builder.add_node("hitl_review", self._node_hitl)
        builder.add_node("finalize", self._node_finalize)

        # --- Define edges ---
        builder.set_entry_point("compliance_check")

        # After compliance: if blocked, go straight to finalize; otherwise investigate
        builder.add_conditional_edges(
            "compliance_check",
            self._route_after_compliance,
            {"blocked": "finalize", "continue": "investigate"},
        )

        builder.add_edge("investigate", "context_lookup")
        builder.add_edge("context_lookup", "score_risk")
        builder.add_edge("score_risk", "communicate")

        # After communication: check if HITL review is needed
        builder.add_conditional_edges(
            "communicate",
            self._route_after_communicate,
            {"hitl": "hitl_review", "finalize": "finalize"},
        )

        builder.add_edge("hitl_review", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()

    # --- Node implementations ---

    def _node_compliance(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.firewall.check(txn_state)
        return self._state_to_dict(txn_state)

    def _node_investigate(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.investigator.run(txn_state)
        return self._state_to_dict(txn_state)

    def _node_context(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.context_agent.run(txn_state)
        return self._state_to_dict(txn_state)

    def _node_risk_score(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.risk_scorer.run(txn_state)
        return self._state_to_dict(txn_state)

    def _node_communicate(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.communication.run(txn_state)
        return self._state_to_dict(txn_state)

    def _node_hitl(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        txn_state = self.hitl.review(txn_state)
        return self._state_to_dict(txn_state)

    def _node_finalize(self, state: dict) -> dict:
        txn_state = self._dict_to_state(state)
        # Apply compliance override if exists (and HITL hasn't already decided)
        if not txn_state.hitl_decision:
            txn_state = self.firewall.apply_override(txn_state)
        # If no override and no HITL decision, use risk action
        if not txn_state.final_action:
            txn_state.final_action = txn_state.risk_action or ACTION_APPROVE
            txn_state.final_reasoning = txn_state.risk_reasoning

        # Safety net: enforce threshold-based action if risk_score contradicts final_action
        from agentic_orchestrator.config import (
            RISK_APPROVE_THRESHOLD, RISK_CHALLENGE_THRESHOLD,
            ACTION_CHALLENGE,
        )
        if txn_state.final_action == ACTION_APPROVE and txn_state.risk_score >= RISK_CHALLENGE_THRESHOLD:
            corrected = ACTION_ESCALATE if txn_state.risk_score >= RISK_CHALLENGE_THRESHOLD + 0.1 else ACTION_CHALLENGE
            txn_state.log("Orchestrator", f"⚠️ Safety override: score {txn_state.risk_score:.3f} too high for {txn_state.final_action} → {corrected}")
            txn_state.final_action = corrected
        elif txn_state.final_action == ACTION_APPROVE and txn_state.risk_score >= RISK_APPROVE_THRESHOLD:
            txn_state.log("Orchestrator", f"⚠️ Safety override: score {txn_state.risk_score:.3f} too high for APPROVE → CHALLENGE")
            txn_state.final_action = ACTION_CHALLENGE
        # Run LLM-as-a-Judge evaluation
        if self.judge:
            self.judge.evaluate(txn_state)
        txn_state.log("Orchestrator", f"✅ FINAL DECISION: {txn_state.final_action}")
        return self._state_to_dict(txn_state)

    # --- Routing logic ---

    def _route_after_compliance(self, state: dict) -> Literal["blocked", "continue"]:
        override = state.get("compliance_override")
        if override and override in ("BLOCK",):
            return "blocked"
        return "continue"

    def _route_after_communicate(self, state: dict) -> Literal["hitl", "finalize"]:
        override = state.get("compliance_override")
        hitl_required = state.get("hitl_required", False)
        if override == ACTION_ESCALATE or hitl_required:
            return "hitl"
        return "finalize"

    # --- State conversion helpers ---

    def _dict_to_state(self, d: dict) -> TransactionState:
        state = TransactionState()
        for key, value in d.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    def _state_to_dict(self, state: TransactionState) -> dict:
        return state.to_dict()

    # --- Public API ---

    def process_transaction(self, state: TransactionState) -> TransactionState:
        """Run a transaction through the full agentic pipeline."""
        state.log("Orchestrator", f"Processing transaction {state.txn_id}...")
        initial = self._state_to_dict(state)
        result = self.graph.invoke(initial)
        return self._dict_to_state(result)

