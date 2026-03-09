"""SKTK knowledge -- RAG, chunking, retrieval, grounding."""

from sktk.knowledge.backends import VectorBackend
from sktk.knowledge.backends.ann import ANNBackend
from sktk.knowledge.backends.in_memory import InMemoryKnowledgeBackend
from sktk.knowledge.chunking import Chunk, fixed_size_chunker, sentence_chunker
from sktk.knowledge.grounding import GroundingConfig, GroundingFilter, Queryable
from sktk.knowledge.knowledge_base import KnowledgeBase, TextSource

try:
    from sktk.knowledge.backends.faiss_backend import FaissBackend
except Exception:  # pragma: no cover
    FaissBackend = None  # type: ignore
try:
    from sktk.knowledge.backends.hnsw_backend import HNSWBackend
except Exception:  # pragma: no cover
    HNSWBackend = None  # type: ignore
from sktk.knowledge.retrieval import (
    BM25Index,
    RetrievalConfig,
    RetrievalMode,
    ScoredChunk,
    reciprocal_rank_fusion,
)

__all__ = [
    "ANNBackend",
    "VectorBackend",
    "BM25Index",
    "Chunk",
    "FaissBackend",
    "GroundingConfig",
    "GroundingFilter",
    "HNSWBackend",
    "InMemoryKnowledgeBackend",
    "KnowledgeBase",
    "Queryable",
    "RetrievalConfig",
    "RetrievalMode",
    "ScoredChunk",
    "TextSource",
    "fixed_size_chunker",
    "reciprocal_rank_fusion",
    "sentence_chunker",
]
