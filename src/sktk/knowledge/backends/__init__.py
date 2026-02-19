"""Knowledge backend implementations for vector retrieval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector store backends used by KnowledgeBase.

    All concrete backends (InMemoryKnowledgeBackend, ANNBackend,
    FaissBackend, HNSWBackend) satisfy this interface.
    """

    async def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Replace all stored data with the given chunks and embeddings."""
        ...

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Append new chunks and embeddings to existing data."""
        ...

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        """Return the top-k most similar chunks for a query embedding."""
        ...

    async def count(self) -> int:
        """Return the number of stored chunks."""
        ...

    async def clear(self) -> None:
        """Remove all stored data."""
        ...
