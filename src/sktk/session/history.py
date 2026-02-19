"""Abstract conversation history interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ConversationHistory(ABC):
    """Interface for append/query/fork operations on conversation transcripts."""

    @abstractmethod
    async def append(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the conversation history."""
        ...

    @abstractmethod
    async def get(self, limit: int | None = None, roles: list[str] | None = None) -> list[dict]:
        """Return messages, optionally filtered by role and limited to the N most recent."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all messages from this history."""
        ...

    @abstractmethod
    async def fork(self, session_id: str) -> ConversationHistory:
        """Create an independent copy of this history under a new session ID."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of messages in this history."""
        ...
