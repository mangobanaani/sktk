"""Performance profiler and session replay.

AgentProfiler: tracks timing per agent invocation step.
SessionRecorder: captures full conversation for replay.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileEntry:
    """A single profiling entry."""

    label: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentProfiler:
    """Tracks timing of agent operations.

    Usage:
        profiler = AgentProfiler()
        async with profiler.measure("llm_call"):
            result = await agent.invoke("hello")
        print(profiler.summary())
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: list[ProfileEntry] = []
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    def measure(self, label: str) -> _ProfileContext:
        """Context manager to measure a block's duration."""
        return _ProfileContext(self, label)

    async def record(self, label: str, duration_ms: float, **metadata: Any) -> None:
        """Manually record a timing entry."""
        async with self._lock:
            self._entries.append(
                ProfileEntry(label=label, duration_ms=duration_ms, metadata=metadata)
            )
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    @property
    def entries(self) -> list[ProfileEntry]:
        """Return a shallow copy of recorded entries.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        return list(self._entries)

    def total_ms(self) -> float:
        """Return cumulative duration across all entries.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        return sum(e.duration_ms for e in self._entries)

    def summary(self) -> dict[str, Any]:
        """Get a summary of all profiled operations.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        total = self.total_ms()
        breakdown = {}
        for e in self._entries:
            if e.label not in breakdown:
                breakdown[e.label] = {"count": 0, "total_ms": 0.0}
            breakdown[e.label]["count"] += 1
            breakdown[e.label]["total_ms"] += e.duration_ms

        return {
            "total_ms": total,
            "entries": len(self._entries),
            "breakdown": breakdown,
        }

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def close(self) -> None:
        """Release profiler resources (currently a no-op)."""
        await self.clear()

    async def __aenter__(self) -> AgentProfiler:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class _ProfileContext:
    """Async context manager used by AgentProfiler.measure() to time code blocks."""

    def __init__(self, profiler: AgentProfiler, label: str) -> None:
        self._profiler = profiler
        self._label = label
        self._start = 0.0

    async def __aenter__(self) -> _ProfileContext:
        """Start the timing measurement."""
        self._start = time.monotonic()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the timer and record the elapsed duration."""
        duration = (time.monotonic() - self._start) * 1000
        await self._profiler.record(self._label, duration)


@dataclass
class ReplayEntry:
    """A single entry in a session recording."""

    turn: int
    role: str
    content: str
    agent_name: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionRecorder:
    """Records full conversation for replay and debugging.

    Usage:
        recorder = SessionRecorder()
        await recorder.record_turn("user", "Hello", agent_name="support")
        await recorder.record_turn("assistant", "Hi!", agent_name="support")

        for entry in recorder.replay():
            print(f"[{entry.role}] {entry.content}")
    """

    def __init__(self, max_entries: int = 50_000) -> None:
        self._entries: list[ReplayEntry] = []
        self._turn = 0
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def record_turn(
        self, role: str, content: str, agent_name: str = "", **metadata: Any
    ) -> None:
        """Record a single conversation turn for later replay."""
        async with self._lock:
            self._turn += 1
            self._entries.append(
                ReplayEntry(
                    turn=self._turn,
                    role=role,
                    content=content,
                    agent_name=agent_name,
                    timestamp=time.time(),
                    metadata=metadata,
                )
            )
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def replay(self) -> list[ReplayEntry]:
        """Get all entries for replay.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Use get_transcript() for a lock-protected copy.
        """
        return list(self._entries)

    def replay_from(self, turn: int) -> list[ReplayEntry]:
        """Replay from a specific turn number.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Use get_transcript() for a lock-protected copy.
        """
        return [e for e in self._entries if e.turn >= turn]

    @property
    def turn_count(self) -> int:
        """Return the current turn counter.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        return self._turn

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all recorded entries to a list of dicts.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        return [
            {
                "turn": e.turn,
                "role": e.role,
                "content": e.content,
                "agent_name": e.agent_name,
                "timestamp": e.timestamp,
                "metadata": e.metadata,
            }
            for e in self._entries
        ]

    async def get_transcript(self) -> list[ReplayEntry]:
        """Get a copy of all recorded entries under lock."""
        async with self._lock:
            return list(self._entries)

    async def close(self) -> None:
        """Release recorder resources (currently a no-op)."""
        await self.clear()

    async def __aenter__(self) -> SessionRecorder:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._turn = 0
