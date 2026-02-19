"""E2E tests for QA workflows with RAG pipeline and real Claude API."""

from __future__ import annotations

import hashlib

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.knowledge.chunking import fixed_size_chunker
from sktk.knowledge.grounding import GroundingConfig, GroundingFilter
from sktk.knowledge.knowledge_base import KnowledgeBase, TextSource
from sktk.knowledge.retrieval import RetrievalConfig, RetrievalMode

pytestmark = pytest.mark.e2e


class HashEmbedder:
    """Deterministic embedder using hashing for test reproducibility."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        raw = [b / 255.0 for b in h]
        # Extend or truncate to dim
        while len(raw) < self._dim:
            raw.extend(raw)
        return raw[: self._dim]


CONTEXT_DOC = (
    "The Eiffel Tower is located in Paris, France. "
    "It was built in 1889 for the World's Fair. "
    "It stands 330 meters tall and is made of iron. "
    "Gustave Eiffel's company designed and built the tower."
)


async def test_grounded_qa(claude_provider):
    source = TextSource(content=CONTEXT_DOC, name="eiffel")
    kb = KnowledgeBase(
        sources=[source],
        embedder=HashEmbedder(),
        chunker=fixed_size_chunker(max_words=30),
        retrieval=RetrievalConfig(mode=RetrievalMode.DENSE, top_k=3),
    )
    await kb.build()

    grounding = GroundingFilter(
        source=kb,
        config=GroundingConfig(max_results=3, min_score=0.0),
    )
    agent = SKTKAgent(
        name="qa-agent",
        instructions=("Answer questions using only the provided context. Be concise."),
        filters=[grounding],
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("How tall is the Eiffel Tower?")
    assert "330" in result or "meter" in result.lower()


async def test_qa_without_relevant_context(claude_provider):
    source = TextSource(content=CONTEXT_DOC, name="eiffel")
    kb = KnowledgeBase(
        sources=[source],
        embedder=HashEmbedder(),
        chunker=fixed_size_chunker(max_words=30),
        retrieval=RetrievalConfig(mode=RetrievalMode.DENSE, top_k=3),
    )
    await kb.build()

    # High min_score means grounding filter won't inject context
    grounding = GroundingFilter(
        source=kb,
        config=GroundingConfig(max_results=3, min_score=999.0),
    )
    agent = SKTKAgent(
        name="qa-no-ctx-agent",
        instructions="You are a helpful assistant. Be concise.",
        filters=[grounding],
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("Say hello")
    # Agent should still respond even without grounding context
    assert isinstance(result, str)
    assert len(result) > 0
