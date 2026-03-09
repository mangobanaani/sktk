"""Graph-based workflow engine for complex multi-agent orchestration.

Supports DAG execution, conditional edges, loops, and checkpointing.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sktk.core.types import maybe_await

logger = logging.getLogger(__name__)

Predicate = Callable[[dict[str, Any]], bool | Awaitable[bool]]
NodeFn = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass
class GraphState:
    """Shared state passed between graph nodes."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the state."""
        return copy.deepcopy(self.data)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self.data = copy.deepcopy(snapshot)


@dataclass
class Edge:
    """A directed edge in the workflow graph."""

    source: str
    target: str


class GraphWorkflow:
    """Directed graph workflow with nodes, edges, and conditional branching.

    Usage:
        wf = GraphWorkflow()
        wf.add_node("research", research_fn)
        wf.add_node("analyze", analyze_fn)
        wf.add_node("write", write_fn)
        wf.add_edge("research", "analyze")
        wf.add_conditional_edge(
            "analyze",
            predicate=lambda bb: bb.get("confidence", 0) > 0.8,
            if_true="write",
            if_false="research",
        )
        result = await wf.execute({"task": "Write a report"})
    """

    def __init__(self, max_iterations: int = 50) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, list[Edge]] = {}
        self._conditional_edges: dict[str, tuple[Predicate, str, str]] = {}
        self._entry: str | None = None
        self._max_iterations = max_iterations

    def add_node(self, name: str, fn: NodeFn) -> GraphWorkflow:
        """Add a node to the graph. First node added is the entry point."""
        self._nodes[name] = fn
        if self._entry is None:
            self._entry = name
        return self

    def add_edge(self, source: str, target: str) -> GraphWorkflow:
        """Add an unconditional edge between nodes."""
        for name in (source, target):
            if name not in self._nodes:
                raise ValueError(f"Node '{name}' not found")
        if source in self._conditional_edges:
            raise ValueError(
                f"Node '{source}' already has a conditional edge; "
                "cannot add an unconditional edge from the same source"
            )
        existing = self._edges.get(source, [])
        if existing:
            raise ValueError(
                f"Node '{source}' already has an unconditional edge to "
                f"'{existing[0].target}'. Use add_conditional_edge() for branching."
            )
        self._edges.setdefault(source, []).append(Edge(source=source, target=target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        predicate: Predicate,
        if_true: str,
        if_false: str,
    ) -> GraphWorkflow:
        """Add a conditional edge that branches based on a predicate."""
        for name in (source, if_true, if_false):
            if name not in self._nodes:
                raise ValueError(f"Node '{name}' not found")
        if source in self._edges:
            raise ValueError(
                f"Node '{source}' already has an unconditional edge; "
                "cannot add a conditional edge from the same source"
            )
        if source in self._conditional_edges:
            raise ValueError(
                f"Node '{source}' already has a conditional edge. "
                "Remove it first before adding a new one."
            )
        self._conditional_edges[source] = (predicate, if_true, if_false)
        return self

    def set_entry(self, name: str) -> GraphWorkflow:
        """Override the entry point."""
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        self._entry = name
        return self

    async def execute(
        self,
        initial_state: dict[str, Any] | None = None,
        checkpoint_fn: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> GraphState:
        """Execute the workflow starting from the entry node.

        Node functions receive the state dict by reference. Mutations to the
        dict inside node functions are permitted and take effect immediately.
        The return-value merge (for dict results) is a convenience for nodes
        that prefer a functional style. Keys prefixed with ``__sktk_`` are
        reserved for internal use and will be filtered from node return values.
        """
        ws = GraphState(data=copy.deepcopy(initial_state) if initial_state else {})

        if self._entry is None:
            raise ValueError("No nodes in workflow")

        current: str | None = self._entry
        iteration = 0

        while current is not None and iteration < self._max_iterations:
            iteration += 1
            node_fn = self._nodes.get(current)
            if node_fn is None:
                raise ValueError(f"Node '{current}' not found")

            logger.debug("Executing node '%s' (iteration %d)", current, iteration)

            result = await maybe_await(node_fn(ws.data))

            if isinstance(result, dict):
                for k, v in result.items():
                    if not k.startswith("__sktk_"):
                        ws.data[k] = v
            elif result is not None:
                ws.data[f"__sktk_{current}_result"] = result

            if checkpoint_fn is not None:
                await checkpoint_fn(current, ws.snapshot())

            # Find next node
            current = await self._next_node(current, ws.data)

        if iteration >= self._max_iterations:
            logger.warning(
                "Workflow reached max_iterations (%d); possible cycle",
                self._max_iterations,
            )

        ws.data["__sktk_iterations"] = iteration
        return ws

    async def _next_node(self, current: str, state: dict[str, Any]) -> str | None:
        """Determine the next node based on edges from current."""
        # Check conditional edges first
        if current in self._conditional_edges:
            predicate, if_true, if_false = self._conditional_edges[current]
            result = await maybe_await(predicate(state))
            return if_true if result else if_false

        # Then check regular edges (O(1) lookup by source)
        candidates = self._edges.get(current, [])
        if not candidates:
            return None

        return candidates[0].target

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram of the workflow."""
        lines = ["graph TD"]
        for name in self._nodes:
            lines.append(f"    {name}[{name}]")
        for edge_list in self._edges.values():
            for edge in edge_list:
                lines.append(f"    {edge.source} --> {edge.target}")
        for source, (_, if_true, if_false) in self._conditional_edges.items():
            lines.append(f"    {source} -->|true| {if_true}")
            lines.append(f"    {source} -->|false| {if_false}")
        return "\n".join(lines)
