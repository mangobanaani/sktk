# tests/unit/team/test_topology_viz.py
from sktk.agent.agent import SKTKAgent


def test_visualize_simple_chain():
    a = SKTKAgent(name="researcher", instructions="R.")
    b = SKTKAgent(name="analyst", instructions="A.")
    pipeline = a >> b
    mermaid = pipeline.visualize()
    assert "graph LR" in mermaid
    assert "researcher" in mermaid
    assert "analyst" in mermaid
    assert "-->" in mermaid


def test_visualize_fan_out():
    a = SKTKAgent(name="lead", instructions="L.")
    b = SKTKAgent(name="worker1", instructions="W1.")
    c = SKTKAgent(name="worker2", instructions="W2.")
    pipeline = a >> [b, c]
    mermaid = pipeline.visualize()
    assert "lead" in mermaid
    assert "worker1" in mermaid
    assert "worker2" in mermaid
