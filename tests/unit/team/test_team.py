# tests/unit/team/test_team.py
import pytest

from sktk.agent.agent import SKTKAgent
from sktk.core.events import CompletionEvent
from sktk.core.types import TokenUsage
from sktk.session.session import Session
from sktk.team.strategies import BroadcastStrategy, RoundRobinStrategy
from sktk.team.team import SKTKTeam
from sktk.testing.mocks import MockKernel


@pytest.fixture
def mock_agents():
    agents = []
    for name in ["researcher", "analyst", "writer"]:
        mk = MockKernel()
        mk.expect_chat_completion(responses=[f"{name} says hello"])
        agent = SKTKAgent(name=name, instructions=f"You are {name}.")
        agent.kernel = mk
        agents.append(agent)
    return agents


@pytest.fixture
def session():
    return Session(id="team-test")


@pytest.mark.asyncio
async def test_team_creation(mock_agents, session):
    team = SKTKTeam(
        agents=mock_agents,
        strategy=RoundRobinStrategy(),
        session=session,
        max_rounds=3,
    )
    assert len(team.agents) == 3
    assert team.max_rounds == 3


@pytest.mark.asyncio
async def test_team_run_round_robin(mock_agents, session):
    team = SKTKTeam(
        agents=mock_agents,
        strategy=RoundRobinStrategy(),
        session=session,
        max_rounds=3,
    )
    result = await team.run("Do something")
    assert result is not None


@pytest.mark.asyncio
async def test_team_stream_events(mock_agents, session):
    team = SKTKTeam(
        agents=mock_agents,
        strategy=RoundRobinStrategy(),
        session=session,
        max_rounds=1,
    )
    events = []
    async for event in team.stream("Do something"):
        events.append(event)
    assert len(events) > 0
    assert isinstance(events[-1], CompletionEvent)
    assert events[0].provider is None
    assert events[0].prompt_hash


@pytest.mark.asyncio
async def test_team_broadcast(session):
    agents = []
    for name in ["a", "b"]:
        mk = MockKernel()
        mk.expect_chat_completion(responses=[f"{name} result"])
        agent = SKTKAgent(name=name, instructions=f"{name}.")
        agent.kernel = mk
        agents.append(agent)
    team = SKTKTeam(
        agents=agents,
        strategy=BroadcastStrategy(),
        session=session,
        max_rounds=1,
    )
    result = await team.run("task")
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_team_broadcast_stream(session):
    agents = []
    for name in ["a", "b"]:
        mk = MockKernel()
        mk.expect_chat_completion(responses=[f"{name} result"])
        agent = SKTKAgent(name=name, instructions=f"{name}.")
        agent.kernel = mk
        agents.append(agent)
    team = SKTKTeam(
        agents=agents,
        strategy=BroadcastStrategy(),
        session=session,
        max_rounds=1,
    )
    events = []
    async for event in team.stream("task"):
        events.append(event)
    assert len(events) == 3  # 2 MessageEvents + 1 CompletionEvent
    assert isinstance(events[-1], CompletionEvent)
    assert events[0].agent in ("a", "b")


@pytest.mark.asyncio
async def test_team_sequential_with_session():
    session = Session(id="seq-test")
    mk = MockKernel()
    mk.expect_chat_completion(responses=["result"])
    agent = SKTKAgent(name="solo", instructions="Do.", session=session)
    agent.kernel = mk
    team = SKTKTeam(
        agents=[agent],
        strategy=RoundRobinStrategy(),
        session=session,
        max_rounds=1,
    )
    result = await team.run("task")
    assert result == "result"


@pytest.mark.asyncio
async def test_team_broadcast_composed_strategy(session):
    agents = []
    for name in ["a", "b"]:
        mk = MockKernel()
        mk.expect_chat_completion(responses=[f"{name} result"])
        agent = SKTKAgent(name=name, instructions=f"{name}.")
        agent.kernel = mk
        agents.append(agent)

    strategy = BroadcastStrategy() | RoundRobinStrategy()
    team = SKTKTeam(agents=agents, strategy=strategy, session=session, max_rounds=1)

    result = await team.run("task")
    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_team_broadcast_stream_composed_strategy(session):
    agents = []
    for name in ["a", "b"]:
        mk = MockKernel()
        mk.expect_chat_completion(responses=[f"{name} result"])
        agent = SKTKAgent(name=name, instructions=f"{name}.")
        agent.kernel = mk
        agents.append(agent)

    strategy = BroadcastStrategy() | RoundRobinStrategy()
    team = SKTKTeam(agents=agents, strategy=strategy, session=session, max_rounds=1)

    events = []
    async for event in team.stream("task"):
        events.append(event)

    assert len(events) == 3  # 2 MessageEvents + CompletionEvent
    assert isinstance(events[-1], CompletionEvent)


@pytest.mark.asyncio
async def test_team_stream_uses_agent_usage_metadata_in_events(session):
    class UsageAgent(SKTKAgent):
        def __init__(self, *, usage: TokenUsage, provider: str, **kwargs):
            super().__init__(**kwargs)
            self._usage = usage
            self._provider = provider

        async def _get_response(self, prompt: str, **kwargs):
            self._last_usage = self._usage
            self._last_provider = self._provider
            return f"{self.name} result"

    agents = [
        UsageAgent(
            name="a",
            instructions="a",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            provider="p1",
        ),
        UsageAgent(
            name="b",
            instructions="b",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=3),
            provider="p2",
        ),
    ]

    team = SKTKTeam(
        agents=agents,
        strategy=BroadcastStrategy(),
        session=session,
        max_rounds=1,
    )

    events = []
    async for event in team.stream("task"):
        events.append(event)

    message_events = events[:-1]
    completion = events[-1]
    assert message_events[0].token_usage is not None
    assert message_events[1].token_usage is not None
    assert message_events[0].provider == "p1"
    assert message_events[1].provider == "p2"
    assert message_events[0].token_usage.total_tokens == 3
    assert message_events[1].token_usage.total_tokens == 5
    assert isinstance(completion, CompletionEvent)
    assert completion.total_tokens is not None
    assert completion.total_tokens.total_tokens == 8


@pytest.mark.asyncio
async def test_team_run_falls_back_when_get_all_agents_raises(session):
    class FailingBroadcastStrategy(RoundRobinStrategy):
        def get_all_agents(self, agents):
            raise RuntimeError("boom")

    mk = MockKernel()
    mk.expect_chat_completion(responses=["fallback result"])
    agent = SKTKAgent(name="solo", instructions="Do.")
    agent.kernel = mk

    team = SKTKTeam(
        agents=[agent],
        strategy=FailingBroadcastStrategy(),
        session=session,
        max_rounds=1,
    )

    result = await team.run("task")
    assert result == "fallback result"


@pytest.mark.asyncio
async def test_team_run_breaks_early_when_strategy_returns_none(session):
    class StopStrategy:
        async def next_agent(self, agents, history, task, **kwargs):
            return None

    team = SKTKTeam(
        agents=[],
        strategy=StopStrategy(),
        session=session,
        max_rounds=3,
    )

    result = await team.run("task")
    assert result == "task"


@pytest.mark.asyncio
async def test_team_stream_falls_back_from_broadcast_and_accumulates_usage(session):
    class FallbackSequentialStrategy:
        def __init__(self):
            self.calls = 0

        def get_all_agents(self, agents):
            raise RuntimeError("boom")

        async def next_agent(self, agents, history, task, **kwargs):
            if self.calls == 0:
                self.calls += 1
                return agents[0]
            return None

    class UsageAgent(SKTKAgent):
        async def _get_response(self, prompt: str, **kwargs):
            self._last_usage = TokenUsage(prompt_tokens=3, completion_tokens=4)
            self._last_provider = "provider-x"
            return "stream result"

    agent = UsageAgent(name="solo", instructions="Do.")
    team = SKTKTeam(
        agents=[agent],
        strategy=FallbackSequentialStrategy(),
        session=session,
        max_rounds=3,
    )

    events = []
    async for event in team.stream("task"):
        events.append(event)

    message, completion = events
    assert message.kind == "message"
    assert message.token_usage is not None
    assert message.token_usage.total_tokens == 7
    assert message.provider == "provider-x"
    assert isinstance(completion, CompletionEvent)
    assert completion.total_rounds == 1
    assert completion.result == "stream result"
    assert completion.total_tokens is not None
    assert completion.total_tokens.total_tokens == 7
