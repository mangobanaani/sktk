"""KnowledgeBase -- RAG as a first-class citizen."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Protocol, runtime_checkable

from sktk.core.context import get_context
from sktk.core.events import RetrievalEvent
from sktk.knowledge.backends import VectorBackend
from sktk.knowledge.backends.in_memory import InMemoryKnowledgeBackend
from sktk.knowledge.chunking import Chunk, Chunker
from sktk.knowledge.retrieval import (
    BM25Index,
    RetrievalConfig,
    RetrievalMode,
    ScoredChunk,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers used by the knowledge base."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Source(Protocol):
    """Protocol for ingestible text sources."""

    name: str

    async def load(self) -> str: ...


@dataclass
class TextSource:
    """Simple in-memory text source for quick KB bootstrapping."""

    content: str
    name: str = "inline"

    async def load(self) -> str:
        return self.content


class KnowledgeBase:
    """Orchestrates source ingestion, chunking, indexing, and retrieval."""

    def __init__(
        self,
        sources: list[Source | str],
        embedder: Embedder,
        chunker: Chunker,
        retrieval: RetrievalConfig,
        backend: VectorBackend | None = None,
        backend_name: str | None = None,
        retrieval_callback: Callable[[str, str, int, float], Any] | None = None,
        agent_name: str | None = None,
        retrieval_event_emitter: Callable[[RetrievalEvent], Any] | None = None,
        stopwords: set[str] | None = None,
    ) -> None:
        if embedder is None:
            raise ValueError("KnowledgeBase requires an embedder")
        self._sources = sources
        self._embedder = embedder
        self._chunker = chunker
        self._retrieval = retrieval
        self._backend: VectorBackend = backend or InMemoryKnowledgeBackend()
        self._backend_name = backend_name
        self._bm25: BM25Index | None = None
        self._chunks: list[Chunk] = []
        self._retrieval_callback = retrieval_callback
        self._agent_name = agent_name or "knowledge_base"
        self._retrieval_event_emitter = retrieval_event_emitter
        self._stopwords = {w.lower() for w in stopwords} if stopwords else set()
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(sources={len(self._sources)}, backend={type(self._backend).__name__})"
        )

    async def close(self) -> None:
        """Release knowledge base resources (no-op for in-memory backends)."""

    async def __aenter__(self) -> KnowledgeBase:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def build(self) -> None:
        """Ingest all sources, chunk them, compute embeddings, and build indices."""
        async with self._lock:
            await self._backend.clear()
            all_chunks: list[Chunk] = []

            for source in self._sources:
                if isinstance(source, str):
                    text = source
                    src_name = "string"
                elif hasattr(source, "load"):
                    text = await source.load()
                    src_name = getattr(source, "name", "unknown")
                else:
                    logger.warning("Skipping unrecognized source type: %s", type(source).__name__)
                    continue

                sanitized = self._sanitize(text)
                chunks = self._chunker(sanitized, src_name)
                all_chunks.extend(chunks)

            if not all_chunks:
                self._chunks = []
                return

            # Dedup by content hash
            seen = set()
            deduped: list[Chunk] = []
            for c in all_chunks:
                h = self._hash_text(c.text)
                if h in seen:
                    continue
                seen.add(h)
                deduped.append(c)
            all_chunks = deduped

            self._chunks = all_chunks

            texts = [self._remove_stopwords(c.text) for c in all_chunks]
            embeddings = await self._embedder.embed(texts)
            self._backend = self._ensure_backend(embeddings)
            await self._backend.store(all_chunks, embeddings)

            if self._retrieval.mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
                self._bm25 = BM25Index()
                self._bm25.index(all_chunks)

            logger.info(
                "Built knowledge base: %d chunks from %d sources",
                len(all_chunks),
                len(self._sources),
            )

    async def query(self, query: str) -> list[ScoredChunk]:
        """Retrieve the most relevant chunks for a query using the configured retrieval mode."""
        # Snapshot references under the lock, then release for slow I/O calls.
        # The backend has its own lock; the embedder is stateless for queries.
        async with self._lock:
            embedder = self._embedder
            backend = self._backend
            bm25 = self._bm25
            retrieval = self._retrieval
            retrieval_callback = self._retrieval_callback
            retrieval_event_emitter = self._retrieval_event_emitter
            agent_name = self._agent_name
            stopwords = self._stopwords

        mode = retrieval.mode
        top_k = retrieval.top_k
        # Apply stopword removal to query for embedding (matches index preprocessing)
        search_query = self._remove_stopwords(query) if stopwords else query

        if mode == RetrievalMode.DENSE:
            query_emb = await embedder.embed_query(search_query)
            results = await backend.search(query_emb, top_k=top_k)

        elif mode == RetrievalMode.SPARSE:
            if bm25 is None:
                return []
            results = bm25.search(query, top_k=top_k)

        elif mode == RetrievalMode.HYBRID:
            query_emb = await embedder.embed_query(search_query)
            dense_results = await backend.search(query_emb, top_k=top_k * 2)
            sparse_results = bm25.search(query, top_k=top_k * 2) if bm25 else []
            results = reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)
        else:
            results = []

        if retrieval.reranker and results:
            results = await retrieval.reranker.rerank(query, results)
            results = results[:top_k]

        # TTL filtering
        if retrieval.ttl_seconds is not None:
            now = time.time()
            cutoff = now - retrieval.ttl_seconds
            filtered = []
            for r in results:
                ts = r.chunk.metadata.get("timestamp")
                if ts is None:
                    filtered.append(r)  # no timestamp = keep
                elif isinstance(ts, int | float) and ts >= cutoff:
                    filtered.append(r)
            results = filtered

        if retrieval_callback is not None:
            top_score = results[0].score if results else 0.0
            await _maybe_call(
                retrieval_callback,
                agent_name,
                query,
                len(results),
                top_score,
            )
        if retrieval_event_emitter is not None:
            top_score = results[0].score if results else 0.0
            _ctx = get_context()
            event = RetrievalEvent(
                agent=agent_name,
                query=query,
                chunks_retrieved=len(results),
                top_score=top_score,
                correlation_id=(_ctx.correlation_id if _ctx else ""),
                timestamp=datetime.now(UTC),
            )
            await _maybe_call(retrieval_event_emitter, event)

        return results

    async def chunk_count(self) -> int:
        """Return the total number of indexed chunks."""
        return await self._backend.count()

    async def add_source(self, source: Source | str) -> None:
        """Add a source to the knowledge base.

        Note: If adding chunks causes the backend to be replaced (e.g., from
        InMemory to a vector backend), all existing chunks will be re-embedded.
        For large knowledge bases, pre-configure the correct backend before
        calling build() to avoid this cost.
        """
        async with self._lock:
            new_chunks: list[Chunk] = []
            if isinstance(source, str):
                text = source
                src_name = "string"
            elif hasattr(source, "load"):
                text = await source.load()
                src_name = getattr(source, "name", "unknown")
            else:
                logger.warning("Skipping unrecognized source type: %s", type(source).__name__)
                return
            logger.debug("Adding source %s to knowledge base", src_name)
            chunks = self._chunker(self._sanitize(text), src_name)
            new_chunks.extend(chunks)

            if not new_chunks:
                return

            # Dedup
            existing_hashes = {self._hash_text(c.text) for c in self._chunks}
            new_unique = [c for c in new_chunks if self._hash_text(c.text) not in existing_hashes]
            if not new_unique:
                return

            new_texts = [self._remove_stopwords(c.text) for c in new_unique]
            new_embeddings = await self._embedder.embed(new_texts)
            old_backend = self._backend
            self._backend = self._ensure_backend(new_embeddings)
            if self._backend is not old_backend:
                # Backend was replaced -- re-store ALL chunks so prior data is not lost
                combined = self._chunks + new_unique
                all_texts = [self._remove_stopwords(c.text) for c in combined]
                all_embeddings = await self._embedder.embed(all_texts)
                await self._backend.store(combined, all_embeddings)
                self._chunks = combined
            else:
                await self._backend.add(new_unique, new_embeddings)
                self._chunks.extend(new_unique)

            if self._retrieval.mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
                self._bm25 = BM25Index()
                self._bm25.index(self._chunks)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _sanitize(text: str) -> str:
        _INJECTION_PREFIXES = ("[system]", "<|system|>", "### system")
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                # Strip fence markers but keep content between them
                continue
            if stripped.startswith(_INJECTION_PREFIXES):
                continue
            lines.append(stripped)
        cleaned = " ".join(lines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _remove_stopwords(self, text: str) -> str:
        if not self._stopwords:
            return text
        words = text.split()
        return " ".join(w for w in words if w.lower() not in self._stopwords)

    def _ensure_backend(self, embeddings: list[list[float]]) -> VectorBackend:
        if not embeddings:
            return self._backend or InMemoryKnowledgeBackend()
        if self._backend_name == "faiss":
            from sktk.knowledge import FaissBackend

            if FaissBackend is None:
                raise ImportError("faiss backend requested but faiss is not installed")
            if not isinstance(self._backend, FaissBackend):
                dim = len(embeddings[0]) if embeddings else 0
                self._backend = FaissBackend(dim)
            return self._backend
        if self._backend_name == "hnsw":
            from sktk.knowledge import HNSWBackend

            if HNSWBackend is None:
                raise ImportError("hnsw backend requested but hnswlib is not installed")
            if not isinstance(self._backend, HNSWBackend):
                dim = len(embeddings[0]) if embeddings else 0
                self._backend = HNSWBackend(dim)
            return self._backend
        return self._backend or InMemoryKnowledgeBackend()


async def _maybe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke sync or async callbacks through a unified awaitable interface."""
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        return await result
    return result
