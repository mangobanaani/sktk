# tests/unit/agent/test_emitter.py
import hashlib

import pytest

from sktk.agent.emitter import AgentEventEmitter
from sktk.core.events import CompletionEvent, MessageEvent, ThinkingEvent, ToolCallEvent
from sktk.core.types import TokenUsage
from sktk.observability.events import EventStream


def _make_emitter(
    agent_name: str = "test-agent",
    instructions: str = "You are a test agent.",
    instructions_version: str | None = "v1",
    usage: TokenUsage | None = None,
    provider: str | None = "openai",
) -> tuple[AgentEventEmitter, EventStream]:
    stream = EventStream()
    emitter = AgentEventEmitter(
        agent_name=agent_name,
        instructions=instructions,
        instructions_version=instructions_version,
        event_stream=stream,
        get_usage=lambda: usage or TokenUsage(prompt_tokens=10, completion_tokens=5),
        get_provider=lambda: provider,
    )
    return emitter, stream


@pytest.mark.asyncio
async def test_emit_thinking_creates_thinking_event():
    emitter, stream = _make_emitter(agent_name="analyst")
    await emitter.emit_thinking()
    events = stream.events
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ThinkingEvent)
    assert event.agent == "analyst"


@pytest.mark.asyncio
async def test_emit_tool_call_creates_tool_call_event():
    emitter, stream = _make_emitter()
    await emitter.emit_tool_call(function="search", arguments={"query": "python"})
    events = stream.events
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.function == "search"
    assert event.arguments == {"query": "python"}


@pytest.mark.asyncio
async def test_emit_message_creates_message_event():
    usage = TokenUsage(prompt_tokens=20, completion_tokens=10)
    emitter, stream = _make_emitter(usage=usage, provider="anthropic")
    await emitter.emit_message(content="Hello world")
    events = stream.events
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MessageEvent)
    assert event.content == "Hello world"
    assert event.token_usage == usage
    assert event.provider == "anthropic"


@pytest.mark.asyncio
async def test_emit_completion_creates_completion_event():
    emitter, stream = _make_emitter()
    await emitter.emit_completion(result="done", duration_seconds=1.5)
    events = stream.events
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, CompletionEvent)
    assert event.result == "done"
    assert event.duration_seconds == 1.5


def test_prompt_hash_is_stable_and_sha256():
    emitter1, _ = _make_emitter(instructions="Do analysis.")
    emitter2, _ = _make_emitter(instructions="Do analysis.")
    emitter3, _ = _make_emitter(instructions="Different instructions.")

    hash1 = emitter1._prompt_hash()
    hash2 = emitter2._prompt_hash()
    hash3 = emitter3._prompt_hash()

    # Same instructions produce the same hash
    assert hash1 == hash2
    # Different instructions produce a different hash
    assert hash1 != hash3
    # SHA-256 hex digest is 64 characters
    assert len(hash1) == 64
    # Matches direct hashlib computation
    expected = hashlib.sha256(b"Do analysis.").hexdigest()
    assert hash1 == expected
