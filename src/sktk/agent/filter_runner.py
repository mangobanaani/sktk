"""FilterRunner -- runs filter pipelines on behalf of an agent."""

from __future__ import annotations

from typing import Any

from sktk.agent.filters import AgentFilter, FilterContext, run_chunk_filters, run_filter_pipeline
from sktk.core.errors import GuardrailException
from sktk.core.types import Deny, Modify


def _estimate_tokens(text: str) -> int:
    """Estimate token count from word count (word_count * 1.3)."""
    return int(len(text.split()) * 1.3)


class FilterRunner:
    """Runs filter pipelines for an agent."""

    __slots__ = ("_filters", "_agent_name")

    def __init__(self, filters: list[AgentFilter], agent_name: str) -> None:
        self._filters = filters
        self._agent_name = agent_name

    @property
    def active(self) -> bool:
        """Return True if there are any filters registered."""
        return bool(self._filters)

    async def run_input(self, content: str) -> str:
        """Run input filters. Returns (possibly modified) content.

        Raises :class:`GuardrailException` if a filter denies the input.
        """
        ctx = FilterContext(
            content=content,
            stage="input",
            agent_name=self._agent_name,
            token_count=_estimate_tokens(content),
        )
        result = await run_filter_pipeline(self._filters, ctx)
        if isinstance(result, Deny):
            raise GuardrailException(reason=result.reason, filter_name="input_pipeline")
        if isinstance(result, Modify):
            return result.content
        return content

    async def run_output(self, content: str) -> str:
        """Run output filters. Returns (possibly modified) content.

        Raises :class:`GuardrailException` if a filter denies the output.
        """
        ctx = FilterContext(
            content=content,
            stage="output",
            agent_name=self._agent_name,
            token_count=_estimate_tokens(content),
        )
        result = await run_filter_pipeline(self._filters, ctx)
        if isinstance(result, Deny):
            raise GuardrailException(reason=result.reason, filter_name="output_pipeline")
        if isinstance(result, Modify):
            return result.content
        return content

    async def run_output_chunk(self, chunk: str, accumulated: str) -> None:
        """Run per-chunk output filters.

        Raises :class:`GuardrailException` if a filter denies the chunk.
        """
        chunk_ctx = FilterContext(
            content=chunk,
            stage="output",
            agent_name=self._agent_name,
            metadata={"accumulated": accumulated},
        )
        chunk_result = await run_chunk_filters(self._filters, chunk_ctx)
        if isinstance(chunk_result, Deny):
            raise GuardrailException(
                reason=chunk_result.reason, filter_name="output_chunk_pipeline"
            )

    async def run_function_call(self, name: str, arguments: dict[str, Any]) -> None:
        """Run function-call filters.

        Raises :class:`GuardrailException` if a filter denies the call.
        """
        ctx = FilterContext(
            content=name,
            stage="function_call",
            agent_name=self._agent_name,
            metadata={"function_name": name, "arguments": arguments},
        )
        result = await run_filter_pipeline(self._filters, ctx)
        if isinstance(result, Deny):
            raise GuardrailException(reason=result.reason, filter_name="function_call_pipeline")
