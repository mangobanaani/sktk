"""Lifecycle hooks for agent invocations.

Hooks run at key points in the agent lifecycle: before invocation,
after success, and after errors. Useful for logging, metrics, cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Hook signatures
OnStartHook = Callable[[str, str], Awaitable[None]]  # (agent_name, input)
OnCompleteHook = Callable[[str, str, Any], Awaitable[None]]  # (agent_name, input, output)
OnErrorHook = Callable[[str, str, Exception], Awaitable[None]]  # (agent_name, input, error)


@dataclass
class LifecycleHooks:
    """Collection of lifecycle hooks for an agent.

    Usage:
        hooks = LifecycleHooks()
        hooks.on_start.append(my_start_hook)
        hooks.on_error.append(my_error_hook)

        agent = SKTKAgent(name="a", instructions="...", hooks=hooks)
    """

    on_start: list[OnStartHook] = field(default_factory=list)
    on_complete: list[OnCompleteHook] = field(default_factory=list)
    on_error: list[OnErrorHook] = field(default_factory=list)

    async def fire_start(self, agent_name: str, input_text: str) -> None:
        """Invoke all on_start hooks before agent invocation."""
        for hook in self.on_start:
            try:
                await hook(agent_name, input_text)
            except Exception:
                logger.exception("on_start hook %r failed for agent %r", hook, agent_name)

    async def fire_complete(self, agent_name: str, input_text: str, output: Any) -> None:
        """Invoke all on_complete hooks after successful invocation."""
        for hook in self.on_complete:
            try:
                await hook(agent_name, input_text, output)
            except Exception:
                logger.exception("on_complete hook %r failed for agent %r", hook, agent_name)

    async def fire_error(self, agent_name: str, input_text: str, error: Exception) -> None:
        """Invoke all on_error hooks after a failed invocation."""
        for hook in self.on_error:
            try:
                await hook(agent_name, input_text, error)
            except Exception:
                logger.exception("on_error hook %r failed for agent %r", hook, agent_name)
