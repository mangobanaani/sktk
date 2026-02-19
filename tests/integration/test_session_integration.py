"""Integration tests for Session with real Claude API."""

from __future__ import annotations

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.session.session import Session

pytestmark = pytest.mark.integration


async def test_session_preserves_context(claude_provider):
    session = Session(id="ctx-test")
    agent = SKTKAgent(
        name="session-agent",
        instructions="You are a helpful assistant. Be concise.",
        service=claude_provider,
        session=session,
        timeout=30.0,
    )
    await agent.invoke("My favorite color is blue. Just acknowledge.")
    result = await agent.invoke("What is my favorite color? One word.")
    assert "blue" in result.lower()


async def test_session_records_messages(claude_provider):
    session = Session(id="record-test")
    agent = SKTKAgent(
        name="record-agent",
        instructions="Be concise.",
        service=claude_provider,
        session=session,
        timeout=30.0,
    )
    await agent.invoke("Say ok")
    history = await session.history.get()
    assert len(history) >= 2
    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles


async def test_multi_turn_accumulation(claude_provider):
    session = Session(id="multi-test")
    agent = SKTKAgent(
        name="multi-agent",
        instructions="Be concise.",
        service=claude_provider,
        session=session,
        timeout=30.0,
    )
    await agent.invoke("Say one")
    await agent.invoke("Say two")
    await agent.invoke("Say three")
    history = await session.history.get()
    # Each turn adds a user + assistant message
    assert len(history) == 6


async def test_session_clear(claude_provider):
    session = Session(id="clear-test")
    agent = SKTKAgent(
        name="clear-agent",
        instructions="Be concise.",
        service=claude_provider,
        session=session,
        timeout=30.0,
    )
    await agent.invoke("Say ok")
    assert len(await session.history.get()) > 0
    await session.history.clear()
    assert len(await session.history.get()) == 0
