"""Helpers to emit RetrievalEvents from grounding/KB usage."""

from __future__ import annotations

from datetime import UTC, datetime

from sktk.core.events import RetrievalEvent
from sktk.observability.events import EventStream


async def emit_retrieval(
    event_stream: EventStream, *, agent: str, query: str, chunks: int, top_score: float
) -> None:
    """Emit a typed RetrievalEvent onto the provided event stream."""
    event = RetrievalEvent(
        agent=agent,
        query=query,
        chunks_retrieved=chunks,
        top_score=top_score,
        correlation_id="",
        timestamp=datetime.now(UTC),
    )
    await event_stream.emit(event)
