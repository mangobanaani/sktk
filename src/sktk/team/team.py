"""SKTKTeam -- multi-agent orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from sktk.agent.agent import SKTKAgent
from sktk.core.context import get_context
from sktk.core.events import CompletionEvent, MessageEvent
from sktk.core.types import TokenUsage
from sktk.observability.events import EventStream
from sktk.session.session import Session
from sktk.team.strategies import BroadcastStrategy, CoordinationStrategy

logger = logging.getLogger(__name__)


@dataclass
class SKTKTeam:
    """Executes collaborative workflows across a set of agents."""

    agents: list[SKTKAgent]
    strategy: CoordinationStrategy
    session: Session | None = None
    max_rounds: int = 20
    _event_stream: EventStream = field(default_factory=EventStream)

    def __repr__(self) -> str:
        return f"SKTKTeam(agents={len(self.agents)}, strategy={type(self.strategy).__name__})"

    def _get_broadcast_agents(self) -> list[SKTKAgent] | None:
        """Return agents list if broadcast strategy, else None."""
        if isinstance(self.strategy, BroadcastStrategy):
            return self.agents
        if hasattr(self.strategy, "get_all_agents"):
            try:
                result = self.strategy.get_all_agents(self.agents)
                return result or None
            except Exception:
                return None
        return None

    @staticmethod
    def _make_message_event(
        agent: SKTKAgent,
        result: Any,
        cid: str,
    ) -> MessageEvent:
        """Build a MessageEvent from an agent invocation result."""
        return MessageEvent(
            agent=agent.name,
            role="assistant",
            content=str(result),
            token_usage=getattr(agent, "_last_usage", None),
            correlation_id=cid,
            timestamp=datetime.now(UTC),
            prompt_hash=_hash_prompt(agent.instructions),
            prompt_version=agent.instructions_version,
            provider=getattr(agent, "_last_provider", None),
        )

    @staticmethod
    def _make_completion_event(
        result: Any,
        rounds: int,
        usage: TokenUsage | None,
        duration: float,
        cid: str,
        agent: SKTKAgent | None,
    ) -> CompletionEvent:
        """Build a CompletionEvent summarising a team run."""
        return CompletionEvent(
            result=result,
            total_rounds=rounds,
            total_tokens=usage,
            duration_seconds=duration,
            correlation_id=cid,
            timestamp=datetime.now(UTC),
            prompt_hash=_hash_prompt(agent.instructions) if agent else None,
            prompt_version=agent.instructions_version if agent else None,
            provider=getattr(agent, "_last_provider", None) if agent else None,
        )

    async def run(self, task: str, **kwargs: Any) -> Any:
        """Execute the team on a task using the configured strategy."""
        broadcast_agents = self._get_broadcast_agents()
        if broadcast_agents is not None:
            return await self._run_broadcast(task, agents=broadcast_agents, **kwargs)
        return await self._run_sequential(task, **kwargs)

    async def _run_sequential(self, task: str, **kwargs: Any) -> Any:
        """Run agents one at a time, piping each result to the next."""
        last_result = task
        actual_rounds = 0
        last_agent: SKTKAgent | None = None
        total_usage: TokenUsage | None = None
        start = time.monotonic()
        ctx = get_context()
        cid = ctx.correlation_id if ctx else ""
        for _round_num in range(self.max_rounds):
            agent = await self.strategy.next_agent(
                self.agents,
                self.session.history if self.session else None,
                last_result,
                **kwargs,
            )
            if agent is None:
                break
            last_agent = agent
            actual_rounds += 1
            result = await agent.invoke(last_result)
            if self.session and self.session is not agent.session:
                await self.session.history.append("assistant", str(result))
            usage = getattr(agent, "_last_usage", None)
            if usage is not None:
                total_usage = usage if total_usage is None else total_usage + usage
            msg_event = self._make_message_event(agent, result, cid)
            await self._event_stream.emit(msg_event)
            last_result = str(result)
        duration = time.monotonic() - start
        completion = self._make_completion_event(
            last_result,
            actual_rounds,
            total_usage,
            duration,
            cid,
            last_agent,
        )
        await self._event_stream.emit(completion)
        return last_result

    async def _run_broadcast(
        self, task: str, agents: list[SKTKAgent] | None = None, **kwargs: Any
    ) -> list[Any]:
        """Send the task to all agents concurrently and collect results.

        Uses ``return_exceptions=True`` so that one agent failure does
        not cancel the others.  Exceptions are logged and excluded from
        the returned list.
        """
        agents = agents or self.agents
        tasks = [agent.invoke(task, **kwargs) for agent in agents]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        successful: list[Any] = []
        for agent, result in zip(agents, raw, strict=True):
            if isinstance(result, BaseException):
                logger.error("Agent %s failed during broadcast: %s", agent.name, result)
            else:
                successful.append(result)
        return successful

    async def stream(self, task: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Yield MessageEvents as each agent completes, ending with a CompletionEvent."""
        start = time.monotonic()
        ctx = get_context()
        cid = ctx.correlation_id if ctx else ""
        actual_rounds = 0
        total_usage: TokenUsage | None = None
        last_agent: SKTKAgent | None = None

        result: Any
        broadcast_agents = self._get_broadcast_agents()
        if broadcast_agents is not None:
            raw = await asyncio.gather(
                *[a.invoke(task, **kwargs) for a in broadcast_agents],
                return_exceptions=True,
            )
            actual_rounds = 1
            successful: list[Any] = []
            for agent, agent_result in zip(broadcast_agents, raw, strict=True):
                if isinstance(agent_result, BaseException):
                    logger.error("Agent %s failed during broadcast: %s", agent.name, agent_result)
                    continue
                last_agent = agent
                successful.append(agent_result)
                usage = getattr(agent, "_last_usage", None)
                if usage is not None:
                    total_usage = usage if total_usage is None else total_usage + usage
                msg_event = self._make_message_event(agent, agent_result, cid)
                await self._event_stream.emit(msg_event)
                yield msg_event
            result = successful
        else:
            last_result = task
            for _round_num in range(self.max_rounds):
                next_agent: SKTKAgent | None = await self.strategy.next_agent(
                    self.agents,
                    self.session.history if self.session else None,
                    last_result,
                    **kwargs,
                )
                if next_agent is None:
                    break
                last_agent = next_agent
                actual_rounds += 1
                agent_result = await next_agent.invoke(last_result)
                usage = getattr(next_agent, "_last_usage", None)
                if usage is not None:
                    total_usage = usage if total_usage is None else total_usage + usage
                msg_event = self._make_message_event(next_agent, agent_result, cid)
                await self._event_stream.emit(msg_event)
                yield msg_event
                last_result = str(agent_result)
            result = last_result

        duration = time.monotonic() - start
        completion = self._make_completion_event(
            result,
            actual_rounds,
            total_usage,
            duration,
            cid,
            last_agent,
        )
        await self._event_stream.emit(completion)
        yield completion


def _hash_prompt(instructions: str) -> str:
    """Return a stable hash of the agent instructions."""
    return hashlib.sha256(instructions.encode()).hexdigest()
