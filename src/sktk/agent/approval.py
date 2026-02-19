"""Human-in-the-loop approval patterns.

Provides mechanisms to pause agent execution and wait for
external human approval before proceeding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sktk.agent.audit_helpers import record_guardrail_event
from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Deny, FilterResult
from sktk.observability.audit import AuditTrail

ApprovalCallback = Callable[[str, str, dict[str, Any]], Awaitable[bool]]


@dataclass
class ApprovalRequest:
    """A pending approval request."""

    agent_name: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    approved: bool | None = None


class ApprovalGate:
    """Gate that pauses execution until human approval.

    Usage:
        gate = ApprovalGate()

        # In agent pipeline
        agent = SKTKAgent(name="a", instructions="...", filters=[gate])

        # In another coroutine or callback
        gate.approve()  # or gate.deny("reason")
    """

    def __init__(self, timeout: float = 300.0, *, audit_trail: AuditTrail | None = None) -> None:
        self._timeout = timeout
        self._event = asyncio.Event()
        self._approved: bool | None = None
        self._deny_reason: str = ""
        self._decided: bool = False
        self._pending: ApprovalRequest | None = None
        self._audit_trail = audit_trail
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> ApprovalRequest | None:
        return self._pending

    def approve(self) -> None:
        """Approve the pending request.

        Must be called from the same event loop thread that runs
        ``wait_for_approval()``.  For cross-thread signaling, use
        ``loop.call_soon_threadsafe(gate.approve)``.

        State mutation and event signaling happen together in a single
        synchronous call, which is atomic within the event loop thread.
        """
        self._approved = True
        self._decided = True
        self._event.set()

    def deny(self, reason: str = "Denied by human reviewer") -> None:
        """Deny the pending request.

        Must be called from the same event loop thread that runs
        ``wait_for_approval()``.  For cross-thread signaling, use
        ``loop.call_soon_threadsafe(gate.deny)`` (with
        ``functools.partial`` for the *reason* argument).

        State mutation and event signaling happen together in a single
        synchronous call, which is atomic within the event loop thread.
        """
        self._approved = False
        self._deny_reason = reason
        self._decided = True
        self._event.set()

    async def reset(self) -> None:
        """Reset the gate for reuse."""
        async with self._lock:
            self._event = asyncio.Event()
            self._approved = None
            self._deny_reason = ""
            self._decided = False
            self._pending = None

    async def wait_for_approval(
        self, agent_name: str, action: str, details: dict[str, Any] | None = None
    ) -> bool:
        """Wait for human approval, returns True if approved.

        The lock is held while setting up the pending request state to
        prevent races with ``reset()``.  It is released before the
        blocking ``wait_for`` so that ``approve()``/``deny()`` (which
        are synchronous and only touch the Event) can proceed.
        """
        async with self._lock:
            self._event.clear()
            self._approved = None
            self._deny_reason = ""
            self._decided = False
            self._pending = ApprovalRequest(
                agent_name=agent_name,
                action=action,
                details=details or {},
            )
        request_details = dict(details or {})
        request_details.setdefault("pending_action", action)
        await record_guardrail_event(
            self._audit_trail,
            "approval_requested",
            agent_name,
            metadata=request_details,
            outcome="pending",
        )
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
        except (TimeoutError, asyncio.CancelledError):
            async with self._lock:
                if self._pending is not None:
                    self._pending.approved = False
            await record_guardrail_event(
                self._audit_trail,
                "approval_timeout",
                agent_name,
                metadata=request_details,
                outcome="error",
                extra={"reason": "approval timeout or cancellation"},
            )
            return False
        async with self._lock:
            if self._pending is not None:
                self._pending.approved = self._approved
            approved = self._approved
            deny_reason = self._deny_reason
        audit_extra: dict[str, Any] = {}
        if not approved:
            audit_extra["reason"] = deny_reason or "Denied by human reviewer"
        await record_guardrail_event(
            self._audit_trail,
            "approval_granted" if approved else "approval_denied",
            agent_name,
            metadata=request_details,
            outcome="success" if approved else "denied",
            extra=audit_extra or None,
        )
        return approved or False

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        """Gate function calls pending human approval."""
        fn_name = context.metadata.get("function_name", context.content)
        approved = await self.wait_for_approval(
            agent_name=context.agent_name or "",
            action=fn_name,
            details=context.metadata,
        )
        if approved:
            return Allow()
        return Deny(reason=self._deny_reason or "Human approval denied")


class AutoApprovalFilter:
    """Filter that auto-approves safe functions, gates dangerous ones.

    Usage:
        gate = ApprovalGate()
        auto = AutoApprovalFilter(
            safe_functions=["search", "calculate"],
            gate=gate,
        )
    """

    def __init__(
        self,
        safe_functions: list[str],
        gate: ApprovalGate,
    ) -> None:
        self._safe = set(safe_functions)
        self._gate = gate

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        fn_name = context.metadata.get("function_name", context.content)
        if fn_name in self._safe:
            return Allow()
        return await self._gate.on_function_call(context)
