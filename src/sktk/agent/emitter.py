"""AgentEventEmitter -- emits typed events on behalf of an agent."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sktk.core.context import get_context
from sktk.core.events import CompletionEvent, MessageEvent, ThinkingEvent, ToolCallEvent
from sktk.core.types import TokenUsage
from sktk.observability.events import EventStream


class AgentEventEmitter:
    """Emits typed events on behalf of an agent."""

    __slots__ = (
        "_agent_name",
        "_instructions",
        "_instructions_version",
        "_event_stream",
        "_get_usage",
        "_get_provider",
    )

    def __init__(
        self,
        agent_name: str,
        instructions: str,
        instructions_version: str | None,
        event_stream: EventStream,
        get_usage: Callable[[], TokenUsage | None],
        get_provider: Callable[[], str | None],
    ) -> None:
        self._agent_name = agent_name
        self._instructions = instructions
        self._instructions_version = instructions_version
        self._event_stream = event_stream
        self._get_usage = get_usage
        self._get_provider = get_provider

    async def emit_thinking(self) -> None:
        await self._event_stream.emit(
            ThinkingEvent(
                agent=self._agent_name,
                correlation_id=self._correlation_id(),
                timestamp=datetime.now(UTC),
            )
        )

    async def emit_tool_call(self, function: str, arguments: dict[str, Any]) -> None:
        await self._event_stream.emit(
            ToolCallEvent(
                agent=self._agent_name,
                plugin="tool",
                function=function,
                arguments=arguments,
                correlation_id=self._correlation_id(),
                timestamp=datetime.now(UTC),
            )
        )

    async def emit_message(self, content: str) -> None:
        await self._event_stream.emit(
            MessageEvent(
                agent=self._agent_name,
                role="assistant",
                content=content,
                token_usage=self._get_usage(),
                correlation_id=self._correlation_id(),
                timestamp=datetime.now(UTC),
                prompt_hash=self._prompt_hash(),
                prompt_version=self._instructions_version,
                provider=self._get_provider(),
            )
        )

    async def emit_completion(
        self, result: Any, duration_seconds: float, total_rounds: int = 1
    ) -> None:
        await self._event_stream.emit(
            CompletionEvent(
                result=result,
                total_rounds=total_rounds,
                total_tokens=self._get_usage(),
                duration_seconds=duration_seconds,
                correlation_id=self._correlation_id(),
                timestamp=datetime.now(UTC),
                prompt_hash=self._prompt_hash(),
                prompt_version=self._instructions_version,
                provider=self._get_provider(),
            )
        )

    def _correlation_id(self) -> str:
        ctx = get_context()
        return ctx.correlation_id if ctx else ""

    def _prompt_hash(self) -> str:
        return hashlib.sha256(self._instructions.encode()).hexdigest()
