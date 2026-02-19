from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from sktk.core.events import MessageEvent, ThinkingEvent
from sktk.core.types import TokenUsage
from sktk.testing.assertions import (
    assert_blackboard_has,
    assert_events_emitted,
    assert_history_contains,
)
from sktk.testing.fixtures import test_session as make_session


@pytest.mark.asyncio
async def test_assert_history_contains_passes():
    s = make_session()
    await s.history.append("user", "hello world")
    await assert_history_contains(s, role="user", content_pattern="hello")


@pytest.mark.asyncio
async def test_assert_history_contains_fails():
    s = make_session()
    await s.history.append("user", "hello")
    with pytest.raises(AssertionError, match="No message found"):
        await assert_history_contains(s, role="assistant", content_pattern="hello")


@pytest.mark.asyncio
async def test_assert_blackboard_has_passes():
    class Result(BaseModel):
        value: int

    s = make_session()
    await s.blackboard.set("key", Result(value=42))
    await assert_blackboard_has(s, key="key", expected=Result(value=42))


@pytest.mark.asyncio
async def test_assert_blackboard_has_fails_missing():
    class Result(BaseModel):
        value: int

    s = make_session()
    with pytest.raises(AssertionError, match="not found"):
        await assert_blackboard_has(s, key="missing", expected=Result(value=1))


def test_assert_events_emitted_passes():
    now = datetime.now(UTC)
    events = [
        ThinkingEvent(agent="a", correlation_id="c", timestamp=now),
        MessageEvent(
            agent="a",
            role="assistant",
            content="hi",
            token_usage=TokenUsage(),
            correlation_id="c",
            timestamp=now,
        ),
    ]
    assert_events_emitted(events, [ThinkingEvent, MessageEvent])


def test_assert_events_emitted_fails():
    now = datetime.now(UTC)
    events = [ThinkingEvent(agent="a", correlation_id="c", timestamp=now)]
    with pytest.raises(AssertionError, match="MessageEvent"):
        assert_events_emitted(events, [ThinkingEvent, MessageEvent])


@pytest.mark.asyncio
async def test_assert_blackboard_has_value_mismatch():
    class Result(BaseModel):
        value: int

    s = make_session()
    await s.blackboard.set("key", Result(value=42))
    with pytest.raises(AssertionError, match="expected"):
        await assert_blackboard_has(s, key="key", expected=Result(value=99))
