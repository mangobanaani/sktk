"""E2E tests for multi-agent team workflows with real Claude API."""

from __future__ import annotations

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.core.events import CompletionEvent, MessageEvent
from sktk.session.session import Session
from sktk.team.strategies import BroadcastStrategy, RoundRobinStrategy
from sktk.team.team import SKTKTeam

pytestmark = pytest.mark.e2e


async def test_round_robin_pipeline(claude_provider):
    writer = SKTKAgent(
        name="writer",
        instructions="You are a writer. Write one short sentence about the given topic.",
        service=claude_provider,
        timeout=30.0,
    )
    editor = SKTKAgent(
        name="editor",
        instructions=(
            "You are an editor. Take the text you receive and make it more concise. "
            "Return only the edited text."
        ),
        service=claude_provider,
        timeout=30.0,
    )
    team = SKTKTeam(
        agents=[writer, editor],
        strategy=RoundRobinStrategy(max_cycles=1),
    )
    result = await team.run("the moon")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


async def test_broadcast_collects_all(claude_provider):
    agent_a = SKTKAgent(
        name="agent-a",
        instructions=(
            "You must always reply with the single word ALPHA. No other text. Just: ALPHA"
        ),
        service=claude_provider,
        timeout=30.0,
    )
    agent_b = SKTKAgent(
        name="agent-b",
        instructions=("You must always reply with the single word BETA. No other text. Just: BETA"),
        service=claude_provider,
        timeout=30.0,
    )
    team = SKTKTeam(
        agents=[agent_a, agent_b],
        strategy=BroadcastStrategy(),
    )
    results = await team.run("Reply now.")
    assert isinstance(results, list)
    assert len(results) == 2
    # Each agent should produce a non-empty result
    assert all(len(r.strip()) > 0 for r in results)


async def test_team_stream_yields_events(claude_provider):
    agent = SKTKAgent(
        name="stream-agent",
        instructions="Be concise.",
        service=claude_provider,
        timeout=30.0,
    )
    team = SKTKTeam(
        agents=[agent],
        strategy=RoundRobinStrategy(max_cycles=1),
    )
    events = []
    async for event in team.stream("Say ok"):
        events.append(event)

    event_types = {type(e) for e in events}
    assert MessageEvent in event_types
    assert CompletionEvent in event_types


async def test_team_with_session(claude_provider):
    session = Session(id="team-session-test")
    agent_a = SKTKAgent(
        name="team-a",
        instructions="Be concise.",
        service=claude_provider,
        timeout=30.0,
    )
    agent_b = SKTKAgent(
        name="team-b",
        instructions="Be concise.",
        service=claude_provider,
        timeout=30.0,
    )
    team = SKTKTeam(
        agents=[agent_a, agent_b],
        strategy=RoundRobinStrategy(max_cycles=1),
        session=session,
    )
    await team.run("Say hello")
    history = await session.history.get()
    # Team session should have recorded assistant messages
    assert len(history) > 0
    roles = [m["role"] for m in history]
    assert "assistant" in roles
