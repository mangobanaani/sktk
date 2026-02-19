# tests/unit/team/test_strategies.py
import pytest
from pydantic import BaseModel

from sktk.agent.agent import SKTKAgent
from sktk.agent.capabilities import Capability
from sktk.session.session import Session
from sktk.team.strategies import (
    BroadcastStrategy,
    CapabilityRoutingStrategy,
    ComposedStrategy,
    RoundRobinStrategy,
)


class AnalysisRequest(BaseModel):
    query: str


@pytest.fixture
def agents():
    return [
        SKTKAgent(name="a", instructions="A."),
        SKTKAgent(name="b", instructions="B."),
        SKTKAgent(name="c", instructions="C."),
    ]


@pytest.fixture
def session():
    return Session(id="test")


@pytest.mark.asyncio
async def test_round_robin_cycles(agents, session):
    strategy = RoundRobinStrategy(max_cycles=0)  # unlimited cycling
    results = []
    for _ in range(6):
        agent = await strategy.next_agent(agents, session.history, "task")
        results.append(agent.name)
    assert results == ["a", "b", "c", "a", "b", "c"]


@pytest.mark.asyncio
async def test_round_robin_reset(agents, session):
    strategy = RoundRobinStrategy()
    await strategy.next_agent(agents, session.history, "task")
    await strategy.reset()
    agent = await strategy.next_agent(agents, session.history, "task")
    assert agent.name == "a"


@pytest.mark.asyncio
async def test_broadcast_returns_none(agents, session):
    strategy = BroadcastStrategy()
    result = await strategy.next_agent(agents, session.history, "task")
    assert result is None


@pytest.mark.asyncio
async def test_broadcast_get_all(agents, session):
    strategy = BroadcastStrategy()
    result = strategy.get_all_agents(agents)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_capability_routing_matches(session):
    agent_a = SKTKAgent(
        name="analyst",
        instructions="Analyze.",
        capabilities=[
            Capability(
                name="analysis", description="", input_types=[AnalysisRequest], output_types=[]
            )
        ],
    )
    agent_b = SKTKAgent(name="writer", instructions="Write.", capabilities=[])
    strategy = CapabilityRoutingStrategy()
    result = await strategy.next_agent(
        [agent_a, agent_b],
        session.history,
        "task",
        input_type=AnalysisRequest,
    )
    assert result is not None
    assert result.name == "analyst"


@pytest.mark.asyncio
async def test_capability_routing_no_match(session):
    agent_a = SKTKAgent(name="writer", instructions="Write.", capabilities=[])
    strategy = CapabilityRoutingStrategy()
    result = await strategy.next_agent(
        [agent_a],
        session.history,
        "task",
        input_type=AnalysisRequest,
    )
    assert result is None


@pytest.mark.asyncio
async def test_strategy_composition(agents, session):
    cap_strategy = CapabilityRoutingStrategy()
    rr_strategy = RoundRobinStrategy()
    composed = cap_strategy | rr_strategy
    result = await composed.next_agent(agents, session.history, "task")
    assert result is not None
    assert result.name == "a"


@pytest.mark.asyncio
async def test_round_robin_empty_agents(session):
    strategy = RoundRobinStrategy()
    result = await strategy.next_agent([], session.history, "task")
    assert result is None


@pytest.mark.asyncio
async def test_round_robin_or_operator(agents, session):
    rr = RoundRobinStrategy()
    broadcast = BroadcastStrategy()
    composed = rr | broadcast
    from sktk.team.strategies import ComposedStrategy

    assert isinstance(composed, ComposedStrategy)


@pytest.mark.asyncio
async def test_broadcast_or_operator(agents, session):
    broadcast = BroadcastStrategy()
    rr = RoundRobinStrategy()
    composed = broadcast | rr
    from sktk.team.strategies import ComposedStrategy

    assert isinstance(composed, ComposedStrategy)


@pytest.mark.asyncio
async def test_composed_strategy_all_return_none(session):
    from sktk.team.strategies import ComposedStrategy

    class NullStrategy:
        async def next_agent(self, agents, history, task, **kwargs):
            return None

    composed = ComposedStrategy([NullStrategy(), NullStrategy()])
    result = await composed.next_agent([], session.history, "task")
    assert result is None


@pytest.mark.asyncio
async def test_composed_strategy_or_extends(session):
    from sktk.team.strategies import ComposedStrategy

    rr = RoundRobinStrategy()
    composed = ComposedStrategy([rr])
    extended = composed | BroadcastStrategy()
    assert isinstance(extended, ComposedStrategy)


def test_composed_strategy_get_all_agents_handles_errors_and_returns_none():
    class BrokenBroadcast:
        def get_all_agents(self, agents):
            raise RuntimeError("boom")

    class NullBroadcast:
        def get_all_agents(self, agents):
            return None

    composed = ComposedStrategy([BrokenBroadcast(), NullBroadcast()])
    assert composed.get_all_agents(["a", "b"]) is None


def test_composed_strategy_get_all_agents_returns_first_non_none_result():
    composed = ComposedStrategy([RoundRobinStrategy(), BroadcastStrategy()])
    agents = ["a", "b"]
    assert composed.get_all_agents(agents) == agents
