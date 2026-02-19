"""Audit trail for agent actions.

Records every agent action with context for compliance and debugging.
Supports pluggable backends (in-memory, file, custom).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditEntry:
    """A single audit log entry."""

    timestamp: str
    action: str
    agent_name: str
    session_id: str
    user_id: str
    correlation_id: str
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    duration_ms: float = 0.0
    previous_hash: str = ""
    entry_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert this audit entry to a plain dictionary."""
        return asdict(self)


@runtime_checkable
class AuditBackend(Protocol):
    """Protocol for audit trail storage backends."""

    async def write(self, entry: AuditEntry) -> None: ...
    async def query(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]: ...
    async def trim(self, max_entries: int) -> None: ...


class InMemoryAuditBackend:
    """In-memory audit backend for development and testing.

    This backend is designed to be used through :class:`AuditTrail`, which
    provides concurrency protection via its own ``asyncio.Lock``.  Direct
    use of this class from concurrent async tasks is not safe and may lead
    to data races on the internal ``_entries`` list.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    async def write(self, entry: AuditEntry) -> None:
        """Append an audit entry to the in-memory store."""
        self._entries.append(entry)

    async def query(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Return entries matching the given filters, capped at limit."""
        results = self._entries
        if session_id:
            results = [e for e in results if e.session_id == session_id]
        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]
        if action:
            results = [e for e in results if e.action == action]
        return results[-limit:]

    async def trim(self, max_entries: int) -> None:
        """Trim entries to keep only the most recent *max_entries*."""
        self._entries = self._entries[-max_entries:]


class AuditTrail:
    """Records and queries agent actions with tamper-evident hashing.

    Usage:
        audit = AuditTrail()
        await audit.record("invoke", agent_name="analyst", session_id="s1",
                          user_id="u1", correlation_id="c1",
                          details={"input": "query"})
        entries = await audit.query(session_id="s1")
    """

    def __init__(self, backend: AuditBackend | None = None, max_entries: int = 50_000) -> None:
        self._backend = backend or InMemoryAuditBackend()
        self._last_hash = ""
        self._max_entries = max_entries
        self._entry_count = 0
        self._trim_warned = False
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        """Release audit trail resources (no-op for in-memory backends)."""

    async def __aenter__(self) -> AuditTrail:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _compute_hash(self, entry_data: dict[str, Any], previous_hash: str) -> str:
        """Compute a SHA-256 hash chaining this entry to the previous one."""
        payload = json.dumps(entry_data, sort_keys=True, default=str) + previous_hash
        return hashlib.sha256(payload.encode()).hexdigest()

    async def record(
        self,
        action: str,
        agent_name: str,
        session_id: str = "",
        user_id: str = "",
        correlation_id: str = "",
        details: dict[str, Any] | None = None,
        outcome: str = "success",
        duration_ms: float = 0.0,
    ) -> AuditEntry:
        """Record an audit entry with tamper-evident chain hash."""
        async with self._lock:
            entry_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": action,
                "agent_name": agent_name,
                "session_id": session_id,
                "user_id": user_id,
                "correlation_id": correlation_id,
                "details": details or {},
                "outcome": outcome,
                "duration_ms": duration_ms,
            }
            entry_hash = self._compute_hash(entry_data, self._last_hash)
            entry = AuditEntry(
                **entry_data,
                previous_hash=self._last_hash,
                entry_hash=entry_hash,
            )
            self._last_hash = entry_hash
            await self._backend.write(entry)
            self._entry_count += 1
            # Trim with 10% hysteresis to amortize the O(n) cost.
            # We only trim when the count exceeds max_entries by 10%,
            # and trim back to max_entries, so the next trim won't
            # happen for another ~10% insertions.
            trim_threshold = self._max_entries + max(1, self._max_entries // 10)
            if self._entry_count >= trim_threshold and hasattr(self._backend, "trim"):
                if not self._trim_warned:
                    logger.warning(
                        "AuditTrail exceeded max_entries (%d), trimming oldest entries",
                        self._max_entries,
                    )
                    self._trim_warned = True
                await self._backend.trim(self._max_entries)
                self._entry_count = self._max_entries
            return entry

    async def query(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        async with self._lock:
            return await self._backend.query(
                session_id=session_id,
                agent_name=agent_name,
                action=action,
                limit=limit,
            )

    def verify_chain(self, entries: list[AuditEntry]) -> bool:
        """Verify the hash chain integrity of audit entries.

        Requires the complete, consecutive chain of entries. Does not
        work on filtered subsets because each entry's hash depends on
        its immediate predecessor.

        Note: After trim() removes oldest entries, the surviving chain
        can still be verified for internal consistency, but cannot be
        anchored to the original genesis entry. For full tamper-evidence,
        disable trimming (set max_entries high enough) or archive trimmed
        entries externally.
        """
        if not entries:
            return True
        prev_hash = entries[0].previous_hash
        for entry in entries:
            if entry.previous_hash != prev_hash:
                return False
            entry_data = {
                "timestamp": entry.timestamp,
                "action": entry.action,
                "agent_name": entry.agent_name,
                "session_id": entry.session_id,
                "user_id": entry.user_id,
                "correlation_id": entry.correlation_id,
                "details": entry.details,
                "outcome": entry.outcome,
                "duration_ms": entry.duration_ms,
            }
            expected = self._compute_hash(entry_data, prev_hash)
            if entry.entry_hash != expected:
                return False
            prev_hash = entry.entry_hash
        return True
