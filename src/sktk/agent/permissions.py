"""Plugin permission policies for agent function control.

Defines which functions/plugins an agent is allowed to call,
preventing unauthorized actions even if the LLM decides to call them.
"""

from __future__ import annotations

import asyncio
import bisect
from dataclasses import dataclass, field

from sktk.agent.audit_helpers import record_guardrail_event
from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Deny, FilterResult
from sktk.observability.audit import AuditTrail


class PermissionPolicy:
    """Policy defining which functions an agent can call.

    Supports allowlist (only these), denylist (everything except these),
    or both (denylist takes precedence).

    Usage:
        policy = PermissionPolicy(
            allow=["search", "calculate"],
            deny=["delete_database", "send_email"],
        )
        # Use as a filter in agent pipeline
        agent = SKTKAgent(name="safe", instructions="...", filters=[policy])
    """

    def __init__(
        self,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        *,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        self._allow = set(allow) if allow else None
        self._deny = set(deny) if deny else set()
        self._audit_trail = audit_trail

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        """Check if the function call is permitted."""
        fn_name = context.metadata.get("function_name", context.content)

        # Deny list takes precedence even when allow list is set
        if fn_name in self._deny:
            reason = f"Function '{fn_name}' is explicitly denied"
            await record_guardrail_event(
                self._audit_trail,
                "permission_denied",
                context.agent_name or "",
                metadata=context.metadata,
                outcome="denied",
                extra={"function_name": fn_name, "reason": reason},
            )
            return Deny(reason=reason)

        if self._allow is not None:
            if fn_name not in self._allow:
                reason = f"Function '{fn_name}' not in allowed list: {sorted(self._allow)}"
                await record_guardrail_event(
                    self._audit_trail,
                    "permission_denied",
                    context.agent_name or "",
                    metadata=context.metadata,
                    outcome="denied",
                    extra={"function_name": fn_name, "reason": reason},
                )
                return Deny(reason=reason)
            return Allow()

        return Allow()

    def is_allowed(self, function_name: str) -> bool:
        """Quick check if a function is permitted."""
        if function_name in self._deny:
            return False
        if self._allow is not None:
            return function_name in self._allow
        return True


@dataclass
class RateLimitPolicy:
    """Rate limit on agent invocations per time window.

    Tracks call counts and denies when limit is exceeded.
    Uses an asyncio.Lock to prevent concurrent requests from
    bypassing the limit.
    """

    max_calls: int = 100
    window_seconds: float = 60.0
    audit_trail: AuditTrail | None = field(default=None, compare=False, repr=False)
    _calls: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        """Remove call timestamps that fall outside the current window."""
        cutoff = now - self.window_seconds
        idx = bisect.bisect_right(self._calls, cutoff)
        if idx:
            del self._calls[:idx]

    async def on_input(self, context: FilterContext) -> FilterResult:
        import time

        async with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._calls) >= self.max_calls:
                reason = f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s"
                await record_guardrail_event(
                    self.audit_trail,
                    "rate_limit_exceeded",
                    context.agent_name or "",
                    metadata=context.metadata,
                    outcome="denied",
                    extra={"reason": reason},
                )
                return Deny(reason=reason)
            self._calls.append(now)
            return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()
