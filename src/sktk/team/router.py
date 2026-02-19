"""Capability-based task routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from sktk.agent.capabilities import match_capabilities
from sktk.core.errors import NoCapableAgentError

if TYPE_CHECKING:
    from sktk.agent.agent import SKTKAgent


class CapabilityRouter:
    """Selects an agent whose declared capabilities match a task."""

    def __init__(self, agents: list[SKTKAgent]) -> None:
        self._agents = agents

    def route(
        self, *, input_type: type[BaseModel] | None = None, tags: list[str] | None = None
    ) -> SKTKAgent:
        """Return the first agent whose capabilities match the given type or tags."""
        for agent in self._agents:
            if not agent.capabilities:
                continue
            matches = match_capabilities(agent.capabilities, input_type=input_type, tags=tags)
            if matches:
                return agent
        raise NoCapableAgentError(
            task_type=input_type.__name__ if input_type else str(tags),
            available=[a.name for a in self._agents],
        )
