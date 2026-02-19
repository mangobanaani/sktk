"""Optional HNSWLIB backend."""

from __future__ import annotations

import asyncio

try:
    import hnswlib  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("hnswlib is not installed; install with extras 'rag-hnsw'") from e

from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk


class HNSWBackend:
    """HNSWLIB vector index backend for approximate nearest-neighbor search."""

    def __init__(self, dim: int, space: str = "cosine", ef_search: int = 64, M: int = 16) -> None:
        self._dim = dim
        self._ef_search = ef_search
        self._M = M
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=1, ef_construction=100, M=M)
        self._index.set_ef(ef_search)
        self._chunks: list[Chunk] = []
        self._built = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def store(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import numpy as np

        async with self._lock:
            self._chunks = list(chunks)
            index, M, ef = self._index, self._M, self._ef_search
            if not embeddings:
                await asyncio.to_thread(index.init_index, max_elements=1, ef_construction=100, M=M)
                index.set_ef(ef)
                self._built = True
                return
            vecs = np.array(embeddings, dtype="float32")
            ids = list(range(len(vecs)))
            await asyncio.to_thread(
                index.init_index, max_elements=len(vecs), ef_construction=100, M=M
            )
            index.set_ef(ef)
            await asyncio.to_thread(index.add_items, vecs, ids)
            self._built = True

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Append new chunks and embeddings to the existing HNSW index."""
        import numpy as np

        async with self._lock:
            index, M, ef = self._index, self._M, self._ef_search
            if not self._built:
                # First add on a fresh instance -- do a full init_index
                self._chunks = list(chunks)
                vecs = np.array(embeddings, dtype="float32")
                ids = list(range(len(vecs)))
                await asyncio.to_thread(
                    index.init_index, max_elements=max(1, len(vecs)), ef_construction=100, M=M
                )
                index.set_ef(ef)
                if len(vecs):
                    await asyncio.to_thread(index.add_items, vecs, ids)
                self._built = True
                return
            start_id = len(self._chunks)
            self._chunks.extend(chunks)
            vecs = np.array(embeddings, dtype="float32")
            ids = list(range(start_id, start_id + len(chunks)))
            await asyncio.to_thread(index.resize_index, len(self._chunks))
            await asyncio.to_thread(index.add_items, vecs, ids)

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        import numpy as np

        async with self._lock:
            if not self._chunks:
                return []
            effective_k = min(top_k, len(self._chunks))
            if effective_k == 0:
                return []
            q = np.array(query_embedding, dtype="float32")
            labels, distances = await asyncio.to_thread(self._index.knn_query, q, k=effective_k)
            results: list[ScoredChunk] = []
            for idx, dist in zip(labels[0], distances[0], strict=True):
                score = max(0.0, min(1.0, float(1 - dist)))
                results.append(
                    ScoredChunk(chunk=self._chunks[int(idx)], score=score, retrieval_method="hnsw")
                )
            return results

    async def count(self) -> int:
        async with self._lock:
            return len(self._chunks)

    async def clear(self) -> None:
        async with self._lock:
            self._chunks.clear()
            index, M, ef = self._index, self._M, self._ef_search
            await asyncio.to_thread(index.init_index, max_elements=1, ef_construction=100, M=M)
            index.set_ef(ef)
            self._built = False
