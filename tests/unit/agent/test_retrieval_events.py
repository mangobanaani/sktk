# tests/unit/agent/test_retrieval_events.py
import pytest

from sktk.agent.pipeline_events import emit_retrieval
from sktk.core.events import RetrievalEvent
from sktk.observability.events import EventStream


def test_retrieval_event_fields():
    ev = RetrievalEvent(
        agent="kb",
        query="q",
        chunks_retrieved=2,
        top_score=0.9,
        correlation_id="c",
        timestamp=None,
    )
    assert ev.kind == "retrieval"
    assert ev.top_score == 0.9


def test_event_stream_receives_retrieval_sync():
    stream = EventStream()
    ev = RetrievalEvent(
        agent="a", query="q", chunks_retrieved=1, top_score=0.5, correlation_id="c", timestamp=None
    )
    import asyncio

    asyncio.get_event_loop().run_until_complete(stream.emit(ev))
    assert stream.events[0].kind == "retrieval"


@pytest.mark.asyncio
async def test_emit_retrieval_sets_timestamp():
    stream = EventStream()
    await emit_retrieval(stream, agent="kb", query="query", chunks=1, top_score=0.1)
    assert stream.events[0].timestamp is not None
