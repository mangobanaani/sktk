"""Session -- the unit of shared state for multi-agent interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sktk.session.backends.memory import InMemoryBlackboard, InMemoryHistory
from sktk.session.blackboard import Blackboard
from sktk.session.history import ConversationHistory


@dataclass
class Session:
    """Container for per-conversation history, blackboard, and metadata."""

    id: str
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: ConversationHistory = field(default_factory=InMemoryHistory)
    blackboard: Blackboard = field(default_factory=InMemoryBlackboard)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r})"

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Explicitly close session resources."""
        try:
            if hasattr(self.history, "close"):
                await self.history.close()
        finally:
            if hasattr(self.blackboard, "close"):
                await self.blackboard.close()
