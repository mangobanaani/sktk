"""Automatic context grounding via RAG.

Automatically queries a KnowledgeBase on user input and injects
relevant context into the prompt before sending to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, FilterResult, Modify


@runtime_checkable
class Queryable(Protocol):
    """Protocol for anything that can be queried for context."""

    async def query(self, query: str) -> list[Any]: ...


@dataclass
class GroundingConfig:
    """Configuration for automatic grounding."""

    max_results: int = 3
    min_score: float = 0.0
    max_tokens: int = 800
    tokens_per_word: float = 1.3
    context_prefix: str = "\n\n[Relevant context from knowledge base]:\n"
    context_suffix: str = "\n[End of context]\n\n"


class GroundingFilter:
    """Filter that auto-grounds prompts with knowledge base context.

    Works with any Queryable source (KnowledgeBase, custom retriever).
    Results can be ScoredChunk objects or dicts with 'text'/'content' keys.

    Usage:
        grounding = GroundingFilter(source=kb)
        agent = SKTKAgent(name="support", instructions="...", filters=[grounding])
    """

    def __init__(
        self,
        source: Queryable,
        config: GroundingConfig | None = None,
    ) -> None:
        self._source = source
        self._config = config or GroundingConfig()
        self._approx_tokens = lambda text: int(len(text.split()) * self._config.tokens_per_word)

    def _extract_text(self, result: Any) -> str:
        """Extract text from a result (ScoredChunk, dict, or string)."""
        # ScoredChunk with .chunk.text
        if hasattr(result, "chunk") and hasattr(result.chunk, "text"):
            return str(result.chunk.text)
        # Dict with text/content
        if isinstance(result, dict):
            return str(result.get("text") or result.get("content") or "")
        # String
        if isinstance(result, str):
            return result
        return str(result)

    def _sanitize(self, text: str) -> str:
        """Strip common prompt markers and code fences, normalize whitespace."""
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("```", "[system]", "<|system|>", "### system")):
                continue
            lines.append(stripped)
        cleaned = " ".join(lines)
        return " ".join(cleaned.split())

    def _get_score(self, result: Any) -> float:
        """Return the relevance score from a result, defaulting to 1.0."""
        if hasattr(result, "score"):
            return float(result.score)
        if isinstance(result, dict):
            return float(result.get("score", 1.0))
        return 1.0

    async def on_input(self, context: FilterContext) -> FilterResult:
        """Query source and inject context into prompt."""
        results = await self._source.query(context.content)
        results = results[: self._config.max_results]

        if not results:
            return Allow()

        if self._config.min_score > 0:
            results = [r for r in results if self._get_score(r) >= self._config.min_score]

        if not results:
            return Allow()

        context_parts = []
        for r in results:
            text = self._sanitize(self._extract_text(r))
            if text:
                context_parts.append(f"- {text}")

        if not context_parts:
            return Allow()

        # Enforce budget
        budget_tokens = self._config.max_tokens
        selected: list[str] = []
        used = 0
        for part in context_parts:
            tokens = self._approx_tokens(part)  # type: ignore[no-untyped-call]
            if used + tokens > budget_tokens:
                break
            selected.append(part)
            used += tokens

        if not selected:
            return Allow()

        grounded_context = (
            self._config.context_prefix + "\n".join(selected) + self._config.context_suffix
        )

        return Modify(content=grounded_context + context.content)

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return Allow()
