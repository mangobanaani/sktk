"""Pipeline topology DSL -- the >> operator for agent pipelines."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sktk.agent.agent import SKTKAgent

#: Union of all topology node types for type annotations.
AnyTopologyNode = Union["AgentNode", "SequentialNode", "ParallelNode"]


def _normalize_rshift_operand(
    other: AnyTopologyNode | SKTKAgent | list[AnyTopologyNode | SKTKAgent],
) -> AnyTopologyNode:
    """Convert right operand of >> to a topology node.

    Returns the normalised node, or ``NotImplemented`` when *other* is
    not a recognised type.
    """
    if isinstance(other, list):
        from sktk.agent.agent import SKTKAgent as SA

        return ParallelNode([AgentNode(agent=a) if isinstance(a, SA) else a for a in other])
    if isinstance(other, AgentNode | SequentialNode | ParallelNode):
        return other
    from sktk.agent.agent import SKTKAgent as SA

    if isinstance(other, SA):
        return AgentNode(agent=other)
    return NotImplemented  # type: ignore[return-value]


@dataclass
class AgentNode:
    """Leaf node wrapping a single agent."""

    agent: SKTKAgent

    def __rshift__(
        self, other: AnyTopologyNode | SKTKAgent | list[AnyTopologyNode | SKTKAgent]
    ) -> SequentialNode:
        right = _normalize_rshift_operand(other)
        if right is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return SequentialNode(self, right)

    async def run(self, message: str, **kwargs: Any) -> str:
        """Invoke the wrapped agent with the given message."""
        return str(await self.agent.invoke(message, **kwargs))

    def visualize(self) -> str:
        """Return a Mermaid-formatted label for this agent node."""
        return f"    {self.agent.name}"


@dataclass
class SequentialNode:
    """Chain two nodes in sequence: left output feeds right input."""

    left: AnyTopologyNode
    right: AnyTopologyNode

    def __rshift__(
        self, other: AnyTopologyNode | SKTKAgent | list[AnyTopologyNode | SKTKAgent]
    ) -> SequentialNode:
        right = _normalize_rshift_operand(other)
        if right is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return SequentialNode(self, right)

    async def run(self, message: str, **kwargs: Any) -> str:
        """Run left node, then feed its output into the right node."""
        try:
            left_result = await self.left.run(message, **kwargs)
        except Exception:
            logger.error("Sequential pipeline failed at left node")
            raise
        if isinstance(left_result, list):
            left_result = "\n".join(str(r) for r in left_result)
        try:
            return str(await self.right.run(str(left_result), **kwargs))
        except Exception:
            logger.error("Sequential pipeline failed at right node")
            raise

    def visualize(self) -> str:
        """Return a Mermaid graph definition for this pipeline."""
        lines = ["graph LR"]
        self._build_mermaid(lines, counter=[0])
        return "\n".join(lines)

    def _build_mermaid(self, lines: list[str], counter: list[int]) -> str:
        left_id = (
            self.left._build_mermaid(lines, counter)
            if hasattr(self.left, "_build_mermaid")
            else self.left.agent.name  # type: ignore[union-attr]
        )
        right_id = (
            self.right._build_mermaid(lines, counter)
            if hasattr(self.right, "_build_mermaid")
            else self.right.agent.name  # type: ignore[union-attr]
        )
        lines.append(f"    {left_id} --> {right_id}")
        return right_id


@dataclass
class ParallelNode:
    """Fan-out: run all nodes in parallel, collect results."""

    nodes: list[AnyTopologyNode]

    def __rshift__(
        self, other: AnyTopologyNode | SKTKAgent | list[AnyTopologyNode | SKTKAgent]
    ) -> SequentialNode:
        right = _normalize_rshift_operand(other)
        if right is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return SequentialNode(self, right)

    async def run(self, message: str, **kwargs: Any) -> list[Any]:
        """Run all child nodes concurrently and return their results.

        Uses ``return_exceptions=True`` so that one node failure does
        not cancel the others.  Exceptions are logged and excluded from
        the returned list.
        """
        tasks = [node.run(message, **kwargs) for node in self.nodes]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[Any] = []
        for _node, result in zip(self.nodes, raw, strict=True):
            if isinstance(result, BaseException):
                logger.error("Node failed in parallel execution: %s", result)
            else:
                results.append(result)
        return results

    def _build_mermaid(self, lines: list[str], counter: list[int]) -> str:
        names = []
        for node in self.nodes:
            if hasattr(node, "_build_mermaid"):
                names.append(node._build_mermaid(lines, counter))
            elif hasattr(node, "agent"):
                names.append(node.agent.name)  # type: ignore[union-attr]
            else:
                counter[0] += 1
                names.append(f"group_{counter[0]}")
        return " & ".join(names)


class TopologyNode:
    """Factory for creating topology nodes."""

    @staticmethod
    def from_agent(agent: SKTKAgent) -> AgentNode:
        return AgentNode(agent=agent)
