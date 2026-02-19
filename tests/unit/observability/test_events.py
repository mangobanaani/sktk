# tests/unit/observability/test_events.py
from datetime import UTC, datetime

import pytest

from sktk.core.events import MessageEvent, ThinkingEvent
from sktk.core.types import TokenUsage
from sktk.observability.events import EventSink, EventStream


@pytest.mark.asyncio
async def test_event_stream_emit_and_collect():
    stream = EventStream()
    now = datetime.now(UTC)
    event = ThinkingEvent(agent="a", correlation_id="c1", timestamp=now)
    await stream.emit(event)
    assert len(stream.events) == 1
    assert stream.events[0] is event


@pytest.mark.asyncio
async def test_event_stream_iter():
    stream = EventStream()
    now = datetime.now(UTC)
    await stream.emit(ThinkingEvent(agent="a", correlation_id="c1", timestamp=now))
    await stream.emit(
        MessageEvent(
            agent="a",
            role="assistant",
            content="hi",
            token_usage=TokenUsage(),
            correlation_id="c1",
            timestamp=now,
        )
    )
    collected = list(stream)
    assert len(collected) == 2


@pytest.mark.asyncio
async def test_event_stream_with_sink():
    received = []

    class TestSink(EventSink):
        async def send(self, event):
            received.append(event)

    stream = EventStream(sinks=[TestSink()])
    now = datetime.now(UTC)
    await stream.emit(ThinkingEvent(agent="a", correlation_id="c1", timestamp=now))
    assert len(received) == 1


@pytest.mark.asyncio
async def test_event_stream_clear():
    stream = EventStream()
    now = datetime.now(UTC)
    await stream.emit(ThinkingEvent(agent="a", correlation_id="c1", timestamp=now))
    await stream.clear()
    assert len(stream.events) == 0


@pytest.mark.asyncio
async def test_event_stream_trims_oldest_when_exceeding_max_events():
    stream = EventStream(max_events=5)
    now = datetime.now(UTC)
    for i in range(8):
        await stream.emit(
            MessageEvent(
                agent="a",
                role="assistant",
                content=f"msg-{i}",
                token_usage=TokenUsage(),
                correlation_id=f"c{i}",
                timestamp=now,
            )
        )
    assert len(stream.events) == 5
    # oldest three (msg-0, msg-1, msg-2) should have been trimmed
    assert stream.events[0].content == "msg-3"
    assert stream.events[-1].content == "msg-7"
