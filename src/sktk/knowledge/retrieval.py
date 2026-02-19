"""Retrieval pipeline for knowledge bases."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from sktk.knowledge.chunking import Chunk


class RetrievalMode(Enum):
    """Supported retrieval strategies for combining dense/sparse search."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking retrieved chunks."""

    async def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]: ...


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline."""

    mode: RetrievalMode = RetrievalMode.DENSE
    top_k: int = 5
    reranker: Reranker | None = None
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    ttl_seconds: float | None = None


@dataclass
class ScoredChunk:
    """A chunk with a relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"


class BM25Index:
    """Pure Python BM25 implementation for sparse retrieval.

    Note: This class is not internally thread-safe. Callers must ensure
    that index() and search() are not called concurrently. KnowledgeBase
    provides this guarantee by creating new BM25Index instances under its
    lock and publishing them atomically.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._chunks: list[Chunk] = []
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        self._avgdl: float = 0.0
        self._n: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from chunks."""
        self._chunks = list(chunks)
        self._docs = [self._tokenize(c.text) for c in chunks]
        self._n = len(self._docs)
        self._df = {}
        self._tf = []

        total_len = 0
        for doc in self._docs:
            total_len += len(doc)
            seen = set(doc)
            for term in seen:
                self._df[term] = self._df.get(term, 0) + 1
            # Pre-compute term frequencies
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)

        self._avgdl = total_len / self._n if self._n > 0 else 0

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Search the index and return top-k scored chunks."""
        if not self._docs or self._avgdl == 0:
            return []

        query_terms = self._tokenize(query)
        scores: list[float] = []

        for idx, doc in enumerate(self._docs):
            score = 0.0
            doc_len = len(doc)
            term_freq = self._tf[idx]  # Use pre-computed

            for term in query_terms:
                if term not in self._df:
                    continue
                df = self._df[term]
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                tf = term_freq.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        filtered = [
            ScoredChunk(chunk=self._chunks[i], score=s, retrieval_method="sparse")
            for i, s in ranked
            if s > 0
        ]
        return filtered

    def _tokenize(self, text: str) -> list[str]:
        # Extract word-like tokens, lowercase, drop punctuation/markup
        return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int = 5,
) -> list[ScoredChunk]:
    """Merge multiple ranked lists using RRF."""
    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for rank, scored in enumerate(results):
            key = f"{scored.chunk.source}:{scored.chunk.index}"
            # RRF score: 1/(k + rank + 1), summed across all lists for each chunk
            rrf_score = 1.0 / (k + rank + 1)
            chunk_scores[key] = chunk_scores.get(key, 0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = scored

    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        ScoredChunk(chunk=chunk_map[key].chunk, score=score, retrieval_method="hybrid")
        for key, score in ranked
    ]
