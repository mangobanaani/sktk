"""Stub ANN backend with cosine similarity, matching the backend interface."""

from __future__ import annotations

import asyncio

from sktk.knowledge.backends.similarity import cosine_similarity
from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk


class ANNBackend:
    """Approximate Nearest Neighbor backend (stub implementation).

    Intended as a drop-in replacement interface; uses cosine similarity
    here to avoid external dependencies.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        async with self._lock:
            self._chunks = list(chunks)
            self._embeddings = list(embeddings)

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Append new chunks and embeddings to existing data."""
        async with self._lock:
            self._chunks.extend(chunks)
            self._embeddings.extend(embeddings)

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        async with self._lock:
            if not self._chunks:
                return []
            scores = [cosine_similarity(query_embedding, emb) for emb in self._embeddings]
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [
                ScoredChunk(chunk=self._chunks[i], score=s, retrieval_method="ann")
                for i, s in ranked
            ]

    async def count(self) -> int:
        async with self._lock:
            return len(self._chunks)

    async def clear(self) -> None:
        async with self._lock:
            self._chunks.clear()
            self._embeddings.clear()
