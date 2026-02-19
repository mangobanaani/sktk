"""Fallback chains for agent error recovery.

Tries agents in order, falling back to the next when one fails.
Common pattern for production resilience.
"""

from __future__ import annotations

from typing import Any

from sktk.agent.agent import SKTKAgent
from sktk.observability.logging import get_logger

_logger = get_logger("sktk.fallback")


class FallbackChain:
    """Try agents in sequence, falling back on failure.

    Usage:
        primary = SKTKAgent(name="gpt4", ...)
        backup = SKTKAgent(name="gpt35", ...)
        chain = FallbackChain([primary, backup])
        result = await chain.invoke("Hello")
    """

    def __init__(
        self,
        agents: list[SKTKAgent],
        fallback_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        if not agents:
            raise ValueError("FallbackChain requires at least one agent")
        self._agents = agents
        self._fallback_exceptions = fallback_exceptions

    @property
    def agents(self) -> list[SKTKAgent]:
        return list(self._agents)

    async def invoke(self, message: str, **kwargs: Any) -> Any:
        """Try each agent in order until one succeeds."""
        last_error: Exception | None = None
        for agent in self._agents:
            try:
                result = await agent.invoke(message, **kwargs)
                return result
            except self._fallback_exceptions as e:
                _logger.warning(
                    "agent failed, trying next",
                    agent_name=agent.name,
                    exc_info=e,
                )
                last_error = e
        raise last_error  # type: ignore[misc]
