"""Structured event stream for SKTK execution observability."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EventSink(Protocol):
    """Protocol for event sinks (stdout, log, message queue)."""

    async def send(self, event: Any) -> None: ...


class EventStream:
    """Collects typed events and optionally forwards to sinks."""

    def __init__(self, sinks: list[EventSink] | None = None, max_events: int = 10_000) -> None:
        self._events: list[Any] = []
        self._sinks = sinks or []
        self._max_events = max_events
        self._trim_warned = False
        self._lock = asyncio.Lock()

    @property
    def events(self) -> list[Any]:
        """Return a shallow copy of collected events.

        Note: best-effort snapshot not guaranteed consistent under
        concurrent writes. Intended for test/debug use only.
        """
        return list(self._events)

    async def emit(self, event: Any) -> None:
        """Append an event and forward it to all registered sinks."""
        async with self._lock:
            self._events.append(event)
            # Trim with 10% hysteresis to amortize the O(n) slice cost
            trim_threshold = self._max_events + max(1, self._max_events // 10)
            if len(self._events) >= trim_threshold:
                if not self._trim_warned:
                    logger.warning(
                        "EventStream exceeded max_events (%d), trimming oldest events",
                        self._max_events,
                    )
                    self._trim_warned = True
                self._events = self._events[-self._max_events :]
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                await sink.send(event)
            except Exception:
                logger.exception("EventSink %r failed to send event %r", sink, event)

    async def clear(self) -> None:
        """Discard all collected events."""
        async with self._lock:
            self._events.clear()

    def __iter__(self) -> Iterator[Any]:
        """Iterate over a snapshot of collected events.

        Returns an iterator over a shallow copy taken at call time.
        Not concurrency-safe for writes that happen during iteration
        of the returned iterator, but protects against structural
        modification (append/trim) of the internal list.
        """
        return iter(list(self._events))

    def __len__(self) -> int:
        """Return the number of collected events.

        Note: not concurrency-safe. Intended for test/debug use only.
        """
        return len(self._events)
