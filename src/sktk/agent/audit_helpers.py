"""Shared helpers for guardrail audit logging."""

from __future__ import annotations

from typing import Any

from sktk.observability.audit import AuditTrail


async def record_guardrail_event(
    audit_trail: AuditTrail | None,
    action: str,
    agent_name: str,
    metadata: dict[str, Any] | None = None,
    outcome: str = "success",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a guardrail-related event if an audit trail is configured."""
    if audit_trail is None:
        return
    details: dict[str, Any] = dict(metadata or {})
    if extra:
        details.update(extra)
    session_id = str(details.get("session_id") or "")
    user_id = str(details.get("user_id") or "")
    correlation_id = str(details.get("correlation_id") or "")
    await audit_trail.record(
        action=action,
        agent_name=agent_name,
        session_id=session_id,
        user_id=user_id,
        correlation_id=correlation_id,
        details=details,
        outcome=outcome,
    )
