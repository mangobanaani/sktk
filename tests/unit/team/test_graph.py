# tests/unit/team/test_graph.py
"""Comprehensive tests for sktk.team.graph (GraphWorkflow, GraphState)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from sktk.team.graph import GraphState, GraphWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(name: str, update: dict[str, Any] | None = None):
    """Return a sync node function that records its execution and merges *update*."""

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        visited: list[str] = state.setdefault("visited", [])
        visited.append(name)
        return update or {}

    return node_fn


async def async_node(state: dict[str, Any]) -> dict[str, Any]:
    """An async node function for verifying maybe_await path."""
    state.setdefault("visited", []).append("async_node")
    return {"async": True}


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_get_set(self):
        ws = GraphState()
        assert ws.get("missing") is None
        assert ws.get("missing", 42) == 42
        ws.set("key", "value")
        assert ws.get("key") == "value"

    def test_snapshot_is_deep_copy(self):
        ws = GraphState(data={"nested": {"a": 1}})
        snap = ws.snapshot()
        snap["nested"]["a"] = 999
        assert ws.get("nested")["a"] == 1, "snapshot must be a deep copy"

    def test_restore_is_deep_copy(self):
        ws = GraphState()
        source = {"nested": {"b": 2}}
        ws.restore(source)
        source["nested"]["b"] = 999
        assert ws.get("nested")["b"] == 2, "restore must deep-copy the input"

    def test_default_data_is_empty_dict(self):
        ws = GraphState()
        assert ws.data == {}


# ---------------------------------------------------------------------------
# GraphWorkflow -- linear execution
# ---------------------------------------------------------------------------


class TestLinearWorkflow:
    @pytest.mark.asyncio
    async def test_linear_a_b_c(self):
        """A -> B -> C executes in order and returns final state."""
        wf = GraphWorkflow()
        wf.add_node("A", make_node("A"))
        wf.add_node("B", make_node("B"))
        wf.add_node("C", make_node("C"))
        wf.add_edge("A", "B")
        wf.add_edge("B", "C")

        result = await wf.execute({"task": "test"})
        assert result.get("visited") == ["A", "B", "C"]
        assert result.get("task") == "test"
        assert result.get("__sktk_iterations") == 3

    @pytest.mark.asyncio
    async def test_single_node_no_edges(self):
        """A single node with no outgoing edges terminates after executing once."""
        wf = GraphWorkflow()
        wf.add_node("only", make_node("only"))

        result = await wf.execute()
        assert result.get("visited") == ["only"]
        assert result.get("__sktk_iterations") == 1

    @pytest.mark.asyncio
    async def test_end_node_terminates(self):
        """When execution reaches a node with no outgoing edges, the workflow stops."""
        wf = GraphWorkflow()
        wf.add_node("start", make_node("start"))
        wf.add_node("end", make_node("end"))
        wf.add_edge("start", "end")
        # "end" has no outgoing edges

        result = await wf.execute()
        assert result.get("visited") == ["start", "end"]

    @pytest.mark.asyncio
    async def test_async_node_function(self):
        """Async node functions are awaited correctly via maybe_await."""
        wf = GraphWorkflow()
        wf.add_node("async_node", async_node)

        result = await wf.execute()
        assert result.get("visited") == ["async_node"]
        assert result.get("async") is True


# ---------------------------------------------------------------------------
# GraphWorkflow -- conditional edges
# ---------------------------------------------------------------------------


class TestConditionalEdges:
    @pytest.mark.asyncio
    async def test_conditional_true_branch(self):
        """When predicate is True, execution follows the if_true branch."""
        wf = GraphWorkflow()
        wf.add_node("check", make_node("check", {"score": 0.9}))
        wf.add_node("pass", make_node("pass"))
        wf.add_node("fail", make_node("fail"))
        wf.add_conditional_edge(
            "check",
            predicate=lambda s: s.get("score", 0) > 0.5,
            if_true="pass",
            if_false="fail",
        )

        result = await wf.execute()
        assert result.get("visited") == ["check", "pass"]

    @pytest.mark.asyncio
    async def test_conditional_false_branch(self):
        """When predicate is False, execution follows the if_false branch."""
        wf = GraphWorkflow()
        wf.add_node("check", make_node("check", {"score": 0.2}))
        wf.add_node("pass", make_node("pass"))
        wf.add_node("fail", make_node("fail"))
        wf.add_conditional_edge(
            "check",
            predicate=lambda s: s.get("score", 0) > 0.5,
            if_true="pass",
            if_false="fail",
        )

        result = await wf.execute()
        assert result.get("visited") == ["check", "fail"]

    @pytest.mark.asyncio
    async def test_async_predicate(self):
        """Async predicates are awaited correctly."""

        async def async_pred(state: dict[str, Any]) -> bool:
            return state.get("ready", False)

        wf = GraphWorkflow()
        wf.add_node("gate", make_node("gate", {"ready": True}))
        wf.add_node("yes", make_node("yes"))
        wf.add_node("no", make_node("no"))
        wf.add_conditional_edge("gate", predicate=async_pred, if_true="yes", if_false="no")

        result = await wf.execute()
        assert result.get("visited") == ["gate", "yes"]

    def test_conditional_edge_rejected_when_unconditional_exists(self):
        """Adding a conditional edge when an unconditional edge already exists raises."""
        wf = GraphWorkflow()
        wf.add_node("src", make_node("src"))
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        wf.add_edge("src", "a")
        with pytest.raises(ValueError, match="already has an unconditional edge"):
            wf.add_conditional_edge(
                "src",
                predicate=lambda s: True,
                if_true="a",
                if_false="b",
            )

    def test_unconditional_edge_rejected_when_conditional_exists(self):
        """Adding an unconditional edge when a conditional edge already exists raises."""
        wf = GraphWorkflow()
        wf.add_node("src", make_node("src"))
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        wf.add_conditional_edge(
            "src",
            predicate=lambda s: True,
            if_true="a",
            if_false="b",
        )
        with pytest.raises(ValueError, match="already has a conditional edge"):
            wf.add_edge("src", "a")

    def test_duplicate_conditional_edge_raises(self):
        """Adding a second conditional edge from the same source raises."""
        wf = GraphWorkflow()
        wf.add_node("src", make_node("src"))
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        wf.add_node("c", make_node("c"))
        wf.add_conditional_edge(
            "src",
            predicate=lambda s: True,
            if_true="a",
            if_false="b",
        )
        with pytest.raises(ValueError, match="already has a conditional edge"):
            wf.add_conditional_edge(
                "src",
                predicate=lambda s: False,
                if_true="b",
                if_false="c",
            )


# ---------------------------------------------------------------------------
# GraphWorkflow -- loops and max_iterations
# ---------------------------------------------------------------------------


class TestLoopsAndMaxIterations:
    @pytest.mark.asyncio
    async def test_loop_terminates_at_max_iterations(self):
        """A cycle (A -> B -> A) should terminate when max_iterations is reached."""
        wf = GraphWorkflow(max_iterations=6)
        wf.add_node("A", make_node("A"))
        wf.add_node("B", make_node("B"))
        wf.add_edge("A", "B")
        wf.add_edge("B", "A")

        result = await wf.execute()
        assert result.get("__sktk_iterations") == 6
        assert result.get("visited") == ["A", "B", "A", "B", "A", "B"]

    @pytest.mark.asyncio
    async def test_loop_emits_warning(self, caplog):
        """When max_iterations is reached, a warning is logged."""
        wf = GraphWorkflow(max_iterations=2)
        wf.add_node("A", make_node("A"))
        wf.add_node("B", make_node("B"))
        wf.add_edge("A", "B")
        wf.add_edge("B", "A")

        # Ensure propagation is enabled so caplog can capture messages
        # (may be disabled by earlier tests calling configure_structured_logging)
        sktk_logger = logging.getLogger("sktk")
        orig_propagate = sktk_logger.propagate
        sktk_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="sktk.team.graph"):
                result = await wf.execute()
        finally:
            sktk_logger.propagate = orig_propagate

        assert result.get("__sktk_iterations") == 2
        assert any("max_iterations" in msg for msg in caplog.messages)

    @pytest.mark.asyncio
    async def test_conditional_loop_exits_early(self):
        """A conditional loop that flips its condition exits before max_iterations."""
        call_count = 0

        def counting_node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            state.setdefault("visited", []).append("counter")
            return {"count": call_count}

        wf = GraphWorkflow(max_iterations=100)
        wf.add_node("counter", counting_node)
        wf.add_node("done", make_node("done"))
        wf.add_conditional_edge(
            "counter",
            predicate=lambda s: s.get("count", 0) < 3,
            if_true="counter",
            if_false="done",
        )

        result = await wf.execute()
        assert result.get("count") == 3
        assert result.get("visited") == ["counter", "counter", "counter", "done"]


# ---------------------------------------------------------------------------
# GraphWorkflow -- validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_add_conditional_edge_missing_source(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        with pytest.raises(ValueError, match="not found"):
            wf.add_conditional_edge("missing", predicate=lambda s: True, if_true="a", if_false="b")

    def test_add_conditional_edge_missing_if_true(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        with pytest.raises(ValueError, match="not found"):
            wf.add_conditional_edge("a", predicate=lambda s: True, if_true="missing", if_false="b")

    def test_add_conditional_edge_missing_if_false(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        with pytest.raises(ValueError, match="not found"):
            wf.add_conditional_edge("a", predicate=lambda s: True, if_true="b", if_false="missing")

    @pytest.mark.asyncio
    async def test_execute_empty_workflow_raises(self):
        wf = GraphWorkflow()
        with pytest.raises(ValueError, match="No nodes"):
            await wf.execute()

    @pytest.mark.asyncio
    async def test_execute_missing_node_in_edge_raises(self):
        """If an edge points to a node that was never added, execute raises."""
        wf = GraphWorkflow()
        wf.add_node("A", make_node("A"))
        # Manually inject a bad edge into the dict-based edge index
        from sktk.team.graph import Edge

        wf._edges.setdefault("A", []).append(Edge(source="A", target="ghost"))

        with pytest.raises(ValueError, match="'ghost' not found"):
            await wf.execute()

    def test_set_entry_missing_node_raises(self):
        wf = GraphWorkflow()
        with pytest.raises(ValueError, match="not found"):
            wf.set_entry("missing")

    def test_set_entry_overrides_default(self):
        wf = GraphWorkflow()
        wf.add_node("first", make_node("first"))
        wf.add_node("second", make_node("second"))
        wf.set_entry("second")
        assert wf._entry == "second"

    def test_first_node_is_default_entry(self):
        wf = GraphWorkflow()
        wf.add_node("alpha", make_node("alpha"))
        wf.add_node("beta", make_node("beta"))
        assert wf._entry == "alpha"

    def test_add_node_returns_self_for_chaining(self):
        wf = GraphWorkflow()
        assert wf.add_node("a", make_node("a")) is wf

    def test_add_edge_returns_self_for_chaining(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        assert wf.add_edge("a", "b") is wf

    def test_add_conditional_edge_returns_self_for_chaining(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        wf.add_node("c", make_node("c"))
        result = wf.add_conditional_edge("a", predicate=lambda s: True, if_true="b", if_false="c")
        assert result is wf

    def test_duplicate_unconditional_edge_from_same_source_raises(self):
        """Adding a second unconditional edge from the same source raises."""
        wf = GraphWorkflow()
        wf.add_node("src", make_node("src"))
        wf.add_node("a", make_node("a"))
        wf.add_node("b", make_node("b"))
        wf.add_edge("src", "a")
        with pytest.raises(ValueError, match="already has an unconditional edge"):
            wf.add_edge("src", "b")

    def test_add_edge_missing_source_raises(self):
        wf = GraphWorkflow()
        wf.add_node("b", make_node("b"))
        with pytest.raises(ValueError, match="'a' not found"):
            wf.add_edge("a", "b")

    def test_add_edge_missing_target_raises(self):
        wf = GraphWorkflow()
        wf.add_node("a", make_node("a"))
        with pytest.raises(ValueError, match="'b' not found"):
            wf.add_edge("a", "b")

    def test_set_entry_returns_self_for_chaining(self):
        wf = GraphWorkflow()
        wf.add_node("x", make_node("x"))
        assert wf.set_entry("x") is wf


# ---------------------------------------------------------------------------
# GraphWorkflow -- to_mermaid
# ---------------------------------------------------------------------------


class TestToMermaid:
    def test_mermaid_contains_graph_header(self):
        wf = GraphWorkflow()
        wf.add_node("A", make_node("A"))
        diagram = wf.to_mermaid()
        assert diagram.startswith("graph TD")

    def test_mermaid_shows_nodes(self):
        wf = GraphWorkflow()
        wf.add_node("research", make_node("research"))
        wf.add_node("write", make_node("write"))
        diagram = wf.to_mermaid()
        assert "research[research]" in diagram
        assert "write[write]" in diagram

    def test_mermaid_shows_regular_edges(self):
        wf = GraphWorkflow()
        wf.add_node("A", make_node("A"))
        wf.add_node("B", make_node("B"))
        wf.add_edge("A", "B")
        diagram = wf.to_mermaid()
        assert "A --> B" in diagram

    def test_mermaid_shows_conditional_edges(self):
        wf = GraphWorkflow()
        wf.add_node("check", make_node("check"))
        wf.add_node("yes", make_node("yes"))
        wf.add_node("no", make_node("no"))
        wf.add_conditional_edge("check", predicate=lambda s: True, if_true="yes", if_false="no")
        diagram = wf.to_mermaid()
        assert "check -->|true| yes" in diagram
        assert "check -->|false| no" in diagram

    def test_mermaid_empty_workflow(self):
        wf = GraphWorkflow()
        diagram = wf.to_mermaid()
        assert diagram == "graph TD"


# ---------------------------------------------------------------------------
# GraphWorkflow -- checkpoint_fn integration
# ---------------------------------------------------------------------------


class TestCheckpointFn:
    @pytest.mark.asyncio
    async def test_checkpoint_fn_called_after_each_node(self):
        """checkpoint_fn receives the node name and a snapshot after each execution."""
        checkpoints: list[tuple[str, dict[str, Any]]] = []

        async def record_checkpoint(node: str, state: dict[str, Any]) -> None:
            checkpoints.append((node, state))

        wf = GraphWorkflow()
        wf.add_node("A", make_node("A"))
        wf.add_node("B", make_node("B"))
        wf.add_edge("A", "B")

        await wf.execute(checkpoint_fn=record_checkpoint)

        assert len(checkpoints) == 2
        assert checkpoints[0][0] == "A"
        assert checkpoints[1][0] == "B"
        # State should reflect cumulative execution
        assert checkpoints[0][1]["visited"] == ["A"]
        assert checkpoints[1][1]["visited"] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_checkpoint_snapshot_is_independent(self):
        """Each checkpoint snapshot is a deep copy, not a mutable reference."""
        checkpoints: list[dict[str, Any]] = []

        async def record(node: str, state: dict[str, Any]) -> None:
            checkpoints.append(state)

        wf = GraphWorkflow()
        wf.add_node("A", make_node("A", {"counter": 1}))
        wf.add_node("B", make_node("B", {"counter": 2}))
        wf.add_edge("A", "B")

        await wf.execute(checkpoint_fn=record)

        # Snapshots should not be mutated by subsequent node execution
        assert checkpoints[0]["counter"] == 1
        assert checkpoints[1]["counter"] == 2


# ---------------------------------------------------------------------------
# GraphWorkflow -- node return value handling
# ---------------------------------------------------------------------------


class TestNodeReturnValues:
    @pytest.mark.asyncio
    async def test_dict_return_merges_into_state(self):
        def returns_dict(state: dict[str, Any]) -> dict[str, Any]:
            return {"merged_key": "merged_value"}

        wf = GraphWorkflow()
        wf.add_node("merger", returns_dict)
        result = await wf.execute()
        assert result.get("merged_key") == "merged_value"

    @pytest.mark.asyncio
    async def test_non_dict_return_stored_under_node_key(self):
        def returns_string(state: dict[str, Any]) -> str:
            return "hello"

        wf = GraphWorkflow()
        wf.add_node("stringer", returns_string)
        result = await wf.execute()
        assert result.get("__sktk_stringer_result") == "hello"

    @pytest.mark.asyncio
    async def test_none_return_does_not_modify_state(self):
        def returns_none(state: dict[str, Any]) -> None:
            pass

        wf = GraphWorkflow()
        wf.add_node("noop", returns_none)
        result = await wf.execute({"existing": True})
        assert result.get("existing") is True
        assert result.get("__sktk_noop_result") is None
