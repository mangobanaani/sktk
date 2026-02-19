"""Typed event dataclasses emitted during SKTK execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sktk.core.types import TokenUsage


@dataclass(frozen=True)
class ThinkingEvent:
    """Emitted when an agent begins a thinking/reasoning step."""

    agent: str
    correlation_id: str
    timestamp: datetime

    @property
    def kind(self) -> str:
        return "thinking"


@dataclass(frozen=True)
class ToolCallEvent:
    """Emitted when an agent invokes a plugin function."""

    agent: str
    plugin: str
    function: str
    arguments: dict[str, Any]
    correlation_id: str
    timestamp: datetime

    @property
    def kind(self) -> str:
        return "tool_call"


@dataclass(frozen=True)
class RetrievalEvent:
    """Emitted when an agent retrieves chunks from a knowledge base."""

    agent: str
    query: str
    chunks_retrieved: int
    top_score: float
    correlation_id: str
    timestamp: datetime

    @property
    def kind(self) -> str:
        return "retrieval"


@dataclass(frozen=True)
class MessageEvent:
    """Emitted when the LLM produces a message response."""

    agent: str
    role: str
    content: str
    token_usage: TokenUsage | None
    correlation_id: str
    timestamp: datetime | None
    prompt_hash: str | None = None
    prompt_version: str | None = None
    provider: str | None = None

    @property
    def kind(self) -> str:
        return "message"


@dataclass(frozen=True)
class CompletionEvent:
    """Emitted when an agent run finishes, carrying aggregated stats."""

    result: Any
    total_rounds: int
    total_tokens: TokenUsage | None
    duration_seconds: float
    correlation_id: str
    timestamp: datetime | None
    prompt_hash: str | None = None
    prompt_version: str | None = None
    provider: str | None = None

    @property
    def kind(self) -> str:
        return "completion"
