"""Coordination strategies for multi-agent teams."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from sktk.agent.capabilities import match_capabilities
from sktk.session.history import ConversationHistory

if TYPE_CHECKING:
    from sktk.agent.agent import SKTKAgent


@runtime_checkable
class CoordinationStrategy(Protocol):
    """Protocol for selecting the next agent in a multi-agent run."""

    async def next_agent(
        self, agents: list[SKTKAgent], history: ConversationHistory | None, task: str, **kwargs: Any
    ) -> SKTKAgent | None: ...


class RoundRobinStrategy:
    """Cycles through agents in order, one per call.

    Set max_cycles to control how many full cycles before returning None.
    Default max_cycles=1 means one pass through all agents.
    Set max_cycles=0 for unlimited cycling (controlled externally via max_rounds).
    """

    def __init__(self, max_cycles: int = 1) -> None:
        self._index = 0
        self._max_cycles = max_cycles
        self._lock: asyncio.Lock = asyncio.Lock()

    async def next_agent(
        self, agents: list[SKTKAgent], history: ConversationHistory | None, task: str, **kwargs: Any
    ) -> SKTKAgent | None:
        """Return the next agent in round-robin order."""
        async with self._lock:
            if not agents:
                return None
            n = len(agents)
            if self._max_cycles > 0 and self._index >= n * self._max_cycles:
                return None
            agent = agents[self._index % n]
            self._index += 1
            # Wrap index modulo n for unlimited cycling to prevent unbounded growth.
            # For bounded cycling the index must reach n * max_cycles to terminate,
            # so we only wrap when max_cycles is 0 (unlimited).
            if self._max_cycles <= 0 and self._index >= n:
                self._index %= n
            return agent

    async def reset(self) -> None:
        async with self._lock:
            self._index = 0

    def __or__(self, other: CoordinationStrategy) -> ComposedStrategy:
        return ComposedStrategy([self, other])


class BroadcastStrategy:
    """Sends the task to all agents in parallel."""

    async def next_agent(
        self, agents: list[SKTKAgent], history: ConversationHistory | None, task: str, **kwargs: Any
    ) -> None:
        """Return None; broadcast delegates to get_all_agents instead."""
        return None

    def get_all_agents(self, agents: list[SKTKAgent]) -> list[SKTKAgent]:
        return list(agents)

    def __or__(self, other: CoordinationStrategy) -> ComposedStrategy:
        return ComposedStrategy([self, other])


class CapabilityRoutingStrategy:
    """Routes to the first agent whose capabilities match the request."""

    async def next_agent(
        self,
        agents: list[SKTKAgent],
        history: ConversationHistory | None,
        task: str,
        *,
        input_type: type[BaseModel] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> SKTKAgent | None:
        """Return the first agent matching the given input_type or tags."""
        if input_type is None and tags is None:
            return None
        for agent in agents:
            if not hasattr(agent, "capabilities") or not agent.capabilities:
                continue
            matches = match_capabilities(agent.capabilities, input_type=input_type, tags=tags)
            if matches:
                return agent
        return None

    def __or__(self, other: CoordinationStrategy) -> ComposedStrategy:
        return ComposedStrategy([self, other])


class ComposedStrategy:
    """Tries multiple strategies in order, returning the first match."""

    def __init__(self, strategies: list[CoordinationStrategy]) -> None:
        self._strategies = strategies

    async def next_agent(
        self, agents: list[SKTKAgent], history: ConversationHistory | None, task: str, **kwargs: Any
    ) -> SKTKAgent | None:
        """Delegate to each strategy in order until one returns an agent."""
        for strategy in self._strategies:
            result = await strategy.next_agent(agents, history, task, **kwargs)
            if result is not None:
                return result
        return None

    def __or__(self, other: CoordinationStrategy) -> ComposedStrategy:
        return ComposedStrategy(self._strategies + [other])

    def get_all_agents(self, agents: list[SKTKAgent]) -> list[SKTKAgent] | None:
        """Return broadcast agent list if any composed strategy supports broadcast.

        Returns ``None`` when no sub-strategy provides a broadcast list,
        or an empty list is returned by all sub-strategies.
        """
        for strategy in self._strategies:
            if hasattr(strategy, "get_all_agents"):
                try:
                    result = strategy.get_all_agents(agents)  # type: ignore[union-attr]
                except Exception:
                    continue
                if result:
                    return result
        return None
