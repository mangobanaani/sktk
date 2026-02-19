"""Optional FAISS backend."""

from __future__ import annotations

import asyncio

try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("faiss is not installed; install with extras 'rag-faiss'") from e

from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk


class FaissBackend:
    """FAISS vector store backend (L2)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatL2(dim)
        self._chunks: list[Chunk] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import numpy as np

        async with self._lock:
            self._chunks = list(chunks)
            vecs = np.array(embeddings, dtype="float32")
            if vecs.shape[1] != self._dim:
                raise ValueError("Embedding dimension mismatch for FAISS backend")
            index = self._index
            await asyncio.to_thread(index.reset)
            await asyncio.to_thread(index.add, vecs)

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Append new chunks and embeddings to the existing FAISS index."""
        import numpy as np

        async with self._lock:
            self._chunks.extend(chunks)
            vecs = np.array(embeddings, dtype="float32")
            if vecs.shape[1] != self._dim:
                raise ValueError("Embedding dimension mismatch for FAISS backend")
            await asyncio.to_thread(self._index.add, vecs)

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        import numpy as np

        async with self._lock:
            if not self._chunks:
                return []
            effective_k = min(top_k, len(self._chunks))
            q = np.array([query_embedding], dtype="float32")
            distances, indices = await asyncio.to_thread(self._index.search, q, effective_k)
            results: list[ScoredChunk] = []
            for idx, dist in zip(indices[0], distances[0], strict=True):
                if idx == -1:
                    continue
                score = 1.0 / (1.0 + float(dist))  # convert L2 distance to similarity in (0, 1]
                results.append(
                    ScoredChunk(chunk=self._chunks[idx], score=score, retrieval_method="faiss")
                )
            return results

    async def count(self) -> int:
        async with self._lock:
            return len(self._chunks)

    async def clear(self) -> None:
        async with self._lock:
            self._chunks.clear()
            await asyncio.to_thread(self._index.reset)
