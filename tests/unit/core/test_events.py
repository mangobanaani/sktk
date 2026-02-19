from datetime import UTC, datetime

from sktk.core.events import (
    CompletionEvent,
    MessageEvent,
    RetrievalEvent,
    ThinkingEvent,
    ToolCallEvent,
)
from sktk.core.types import TokenUsage


def test_thinking_event():
    now = datetime.now(UTC)
    e = ThinkingEvent(agent="analyst", correlation_id="c1", timestamp=now)
    assert e.agent == "analyst"
    assert e.correlation_id == "c1"
    assert e.timestamp == now


def test_tool_call_event():
    e = ToolCallEvent(
        agent="analyst",
        plugin="math",
        function="add",
        arguments={"a": 1, "b": 2},
        correlation_id="c1",
        timestamp=datetime.now(UTC),
    )
    assert e.plugin == "math"
    assert e.function == "add"
    assert e.arguments == {"a": 1, "b": 2}


def test_retrieval_event():
    e = RetrievalEvent(
        agent="researcher",
        query="quantum computing",
        chunks_retrieved=5,
        top_score=0.92,
        correlation_id="c1",
        timestamp=datetime.now(UTC),
    )
    assert e.chunks_retrieved == 5
    assert e.top_score == 0.92


def test_message_event():
    e = MessageEvent(
        agent="writer",
        role="assistant",
        content="Here is the report.",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        correlation_id="c1",
        timestamp=datetime.now(UTC),
        prompt_hash="hash",
        prompt_version="v1",
    )
    assert e.token_usage.total_tokens == 150
    assert e.prompt_hash == "hash"
    assert e.prompt_version == "v1"


def test_completion_event():
    e = CompletionEvent(
        result="final answer",
        total_rounds=3,
        total_tokens=TokenUsage(prompt_tokens=500, completion_tokens=200),
        duration_seconds=12.5,
        correlation_id="c1",
        timestamp=datetime.now(UTC),
    )
    assert e.total_rounds == 3
    assert e.duration_seconds == 12.5


def test_event_kind():
    now = datetime.now(UTC)
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
    assert ThinkingEvent(agent="a", correlation_id="c", timestamp=now).kind == "thinking"
    assert (
        ToolCallEvent(
            agent="a",
            plugin="p",
            function="f",
            arguments={},
            correlation_id="c",
            timestamp=now,
        ).kind
        == "tool_call"
    )
    assert (
        RetrievalEvent(
            agent="a",
            query="q",
            chunks_retrieved=1,
            top_score=0.9,
            correlation_id="c",
            timestamp=now,
        ).kind
        == "retrieval"
    )
    assert (
        MessageEvent(
            agent="a",
            role="assistant",
            content="hi",
            token_usage=usage,
            correlation_id="c",
            timestamp=now,
        ).kind
        == "message"
    )
    assert (
        CompletionEvent(
            result="done",
            total_rounds=1,
            total_tokens=usage,
            duration_seconds=1.0,
            correlation_id="c",
            timestamp=now,
        ).kind
        == "completion"
    )
