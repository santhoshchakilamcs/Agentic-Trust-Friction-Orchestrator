"""
Base agent class for all specialized agents.
"""

from abc import ABC, abstractmethod

from agentic_orchestrator.memory.short_term import TransactionState


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    name: str = "BaseAgent"

    @abstractmethod
    def run(self, state: TransactionState) -> TransactionState:
        """Process the transaction state and return updated state."""
        ...

    def __repr__(self):
        return f"<{self.name}>"
