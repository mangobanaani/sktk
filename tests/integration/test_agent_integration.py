"""Integration tests for SKTKAgent with real Claude API."""

from __future__ import annotations

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.core.events import CompletionEvent, MessageEvent, ThinkingEvent

pytestmark = pytest.mark.integration


async def test_agent_invoke_returns_string(claude_agent):
    result = await claude_agent.invoke("What is 2+2? Reply with just the number.")
    assert "4" in str(result)


async def test_agent_follows_instructions(claude_provider):
    agent = SKTKAgent(
        name="pirate",
        instructions="You are a pirate. Always include 'arrr' in your response.",
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("Greet me.")
    assert "arrr" in result.lower()


async def test_agent_invoke_stream_chunks(claude_agent):
    chunks = []
    async for chunk in claude_agent.invoke_stream("Say hello"):
        chunks.append(chunk)
    assert len(chunks) >= 1
    full = "".join(chunks)
    assert len(full) > 0


async def test_agent_records_usage_metadata(claude_agent):
    await claude_agent.invoke("Say ok")
    assert claude_agent._last_usage is not None
    assert claude_agent._last_usage.prompt_tokens > 0
    assert claude_agent._last_usage.completion_tokens > 0


async def test_agent_emits_events(claude_agent):
    await claude_agent.event_stream.clear()
    await claude_agent.invoke("Say ok")
    events = claude_agent.event_stream.events
    event_types = {type(e) for e in events}
    assert ThinkingEvent in event_types
    assert MessageEvent in event_types
    assert CompletionEvent in event_types


async def test_agent_respects_timeout(claude_provider):
    agent = SKTKAgent(
        name="timeout-test",
        instructions="Be concise.",
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("Say ok")
    assert isinstance(result, str)
    assert len(result) > 0
