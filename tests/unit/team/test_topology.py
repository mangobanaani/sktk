# tests/unit/team/test_topology.py
import pytest

from sktk.agent.agent import SKTKAgent
from sktk.team.topology import (
    AgentNode,
    ParallelNode,
    SequentialNode,
)


@pytest.fixture
def agents():
    a = SKTKAgent(name="researcher", instructions="Research.")
    b = SKTKAgent(name="analyst", instructions="Analyze.")
    c = SKTKAgent(name="writer", instructions="Write.")
    return a, b, c


def test_agent_node_creation(agents):
    a, _, _ = agents
    node = AgentNode(agent=a)
    assert node.agent.name == "researcher"


def test_sequential_node(agents):
    a, b, _ = agents
    node = SequentialNode(AgentNode(a), AgentNode(b))
    assert node.left.agent.name == "researcher"
    assert node.right.agent.name == "analyst"


def test_parallel_node(agents):
    a, b, c = agents
    node = ParallelNode([AgentNode(a), AgentNode(b), AgentNode(c)])
    assert len(node.nodes) == 3


def test_rshift_two_agents(agents):
    a, b, _ = agents
    pipeline = a >> b
    assert isinstance(pipeline, SequentialNode)
    assert pipeline.left.agent.name == "researcher"
    assert pipeline.right.agent.name == "analyst"


def test_rshift_chain_three(agents):
    a, b, c = agents
    pipeline = a >> b >> c
    assert isinstance(pipeline, SequentialNode)


def test_rshift_fan_out(agents):
    a, b, c = agents
    pipeline = a >> [b, c]
    assert isinstance(pipeline, SequentialNode)
    assert isinstance(pipeline.right, ParallelNode)
    assert len(pipeline.right.nodes) == 2


def test_rshift_fan_in(agents):
    a, b, c = agents
    pipeline = a >> [b, c] >> a
    assert isinstance(pipeline, SequentialNode)


def test_agent_node_rshift_to_agent(agents):
    a, b, _ = agents
    node = AgentNode(agent=a)
    result = node >> b
    assert isinstance(result, SequentialNode)
    assert result.left.agent.name == "researcher"
    assert result.right.agent.name == "analyst"


def test_agent_node_rshift_to_list(agents):
    a, b, c = agents
    node = AgentNode(agent=a)
    result = node >> [b, c]
    assert isinstance(result, SequentialNode)
    assert isinstance(result.right, ParallelNode)


def test_agent_node_rshift_to_node(agents):
    a, b, _ = agents
    left = AgentNode(agent=a)
    right = AgentNode(agent=b)
    result = left >> right
    assert isinstance(result, SequentialNode)


def test_agent_node_rshift_not_implemented(agents):
    a, _, _ = agents
    node = AgentNode(agent=a)
    result = node.__rshift__(42)
    assert result is NotImplemented


def test_sequential_node_rshift_to_agent(agents):
    a, b, c = agents
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    result = seq >> c
    assert isinstance(result, SequentialNode)


def test_sequential_node_rshift_to_list(agents):
    a, b, c = agents
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    result = seq >> [b, c]
    assert isinstance(result, SequentialNode)
    assert isinstance(result.right, ParallelNode)


def test_sequential_node_rshift_to_existing_node(agents):
    a, b, c = agents
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    right = AgentNode(c)
    result = seq >> right
    assert isinstance(result, SequentialNode)
    assert result.right is right


def test_sequential_node_rshift_not_implemented(agents):
    a, b, _ = agents
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    result = seq.__rshift__(42)
    assert result is NotImplemented


def test_parallel_node_rshift_to_agent(agents):
    a, b, c = agents
    par = ParallelNode([AgentNode(a), AgentNode(b)])
    result = par >> c
    assert isinstance(result, SequentialNode)


def test_parallel_node_rshift_to_list(agents):
    a, b, c = agents
    par = ParallelNode([AgentNode(a), AgentNode(b)])
    result = par >> [b, c]
    assert isinstance(result, SequentialNode)
    assert isinstance(result.right, ParallelNode)


def test_parallel_node_rshift_to_existing_node(agents):
    a, b, c = agents
    par = ParallelNode([AgentNode(a), AgentNode(b)])
    right = AgentNode(c)
    result = par >> right
    assert isinstance(result, SequentialNode)
    assert result.right is right


def test_parallel_node_rshift_not_implemented(agents):
    a, b, _ = agents
    par = ParallelNode([AgentNode(a), AgentNode(b)])
    result = par.__rshift__(42)
    assert result is NotImplemented


@pytest.mark.asyncio
async def test_agent_node_run(agents):
    a, _, _ = agents
    from sktk.testing.mocks import MockKernel

    mk = MockKernel()
    mk.expect_chat_completion(responses=["hello"])
    a.kernel = mk
    node = AgentNode(agent=a)
    result = await node.run("test")
    assert result == "hello"


@pytest.mark.asyncio
async def test_sequential_node_run(agents):
    a, b, _ = agents
    from sktk.testing.mocks import MockKernel

    mk_a = MockKernel()
    mk_a.expect_chat_completion(responses=["step1"])
    a.kernel = mk_a
    mk_b = MockKernel()
    mk_b.expect_chat_completion(responses=["step2"])
    b.kernel = mk_b
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    result = await seq.run("input")
    assert result == "step2"


@pytest.mark.asyncio
async def test_parallel_node_run(agents):
    from sktk.testing.mocks import MockKernel

    a, b, _ = agents
    mk_a = MockKernel()
    mk_a.expect_chat_completion(responses=["res_a"])
    a.kernel = mk_a
    mk_b = MockKernel()
    mk_b.expect_chat_completion(responses=["res_b"])
    b.kernel = mk_b
    par = ParallelNode([AgentNode(a), AgentNode(b)])
    results = await par.run("input")
    assert results == ["res_a", "res_b"]


def test_agent_node_visualize(agents):
    a, _, _ = agents
    node = AgentNode(agent=a)
    assert "researcher" in node.visualize()


def test_sequential_visualize(agents):
    a, b, _ = agents
    seq = SequentialNode(AgentNode(a), AgentNode(b))
    viz = seq.visualize()
    assert "graph LR" in viz
    assert "researcher" in viz
    assert "analyst" in viz


def test_parallel_build_mermaid_with_subgroups(agents):
    a, b, c = agents
    inner_seq = SequentialNode(AgentNode(a), AgentNode(b))
    par = ParallelNode([inner_seq, AgentNode(c)])
    lines: list[str] = []
    result = par._build_mermaid(lines, counter=[0])
    assert "group_" in result or "writer" in result


def test_topology_node_factory(agents):
    from sktk.team.topology import TopologyNode

    a, _, _ = agents
    node = TopologyNode.from_agent(a)
    assert isinstance(node, AgentNode)
    assert node.agent.name == "researcher"


@pytest.mark.asyncio
async def test_sequential_node_propagates_left_error(agents):
    """SequentialNode logs and re-raises when left node fails."""
    a, b, _ = agents
    left_node = AgentNode(agent=a)
    right_node = AgentNode(agent=b)
    seq = SequentialNode(left_node, right_node)

    # Monkey-patch left node to fail
    async def failing_run(message, **kwargs):
        raise RuntimeError("left exploded")

    left_node.run = failing_run

    with pytest.raises(RuntimeError, match="left exploded"):
        await seq.run("hello")
