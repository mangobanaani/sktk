import pytest

import sktk.knowledge as knowledge_pkg
from sktk.knowledge.backends.in_memory import InMemoryKnowledgeBackend
from sktk.knowledge.chunking import Chunk
from sktk.knowledge.knowledge_base import KnowledgeBase
from sktk.knowledge.retrieval import RetrievalConfig


class DummyEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_default_backend_when_none():
    kb = KnowledgeBase(
        sources=["x"],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
    )
    await kb.build()
    assert isinstance(kb._backend, InMemoryKnowledgeBackend)


@pytest.mark.asyncio
async def test_faiss_backend_selection_if_available():
    try:
        import faiss  # noqa: F401
    except ImportError:
        pytest.skip("faiss not installed")
    kb = KnowledgeBase(
        sources=["x"],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
        backend_name="faiss",
    )
    await kb.build()
    assert kb._backend is not None


@pytest.mark.asyncio
async def test_backend_name_selects_requested_backend(monkeypatch):
    class DummyFaissBackend:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self._count = 0

        async def store(self, chunks, embeddings):
            self._count = len(chunks)

        async def search(self, query_embedding, top_k=5):
            return []

        async def count(self):
            return self._count

        async def clear(self):
            self._count = 0

    monkeypatch.setattr(knowledge_pkg, "FaissBackend", DummyFaissBackend)

    kb = KnowledgeBase(
        sources=["x"],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
        backend_name="faiss",
    )
    await kb.build()

    assert isinstance(kb._backend, DummyFaissBackend)


def test_ensure_backend_raises_when_faiss_unavailable(monkeypatch):
    monkeypatch.setattr(knowledge_pkg, "FaissBackend", None)
    kb = KnowledgeBase(
        sources=[],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
        backend_name="faiss",
    )
    with pytest.raises(ImportError, match="faiss backend requested but faiss is not installed"):
        kb._ensure_backend([[1.0, 0.0]])


def test_ensure_backend_raises_when_hnsw_unavailable(monkeypatch):
    monkeypatch.setattr(knowledge_pkg, "HNSWBackend", None)
    kb = KnowledgeBase(
        sources=[],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
        backend_name="hnsw",
    )
    with pytest.raises(ImportError, match="hnsw backend requested but hnswlib is not installed"):
        kb._ensure_backend([[1.0, 0.0]])


def test_ensure_backend_selects_hnsw_when_available(monkeypatch):
    class DummyHNSWBackend:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    monkeypatch.setattr(knowledge_pkg, "HNSWBackend", DummyHNSWBackend)
    kb = KnowledgeBase(
        sources=[],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
        backend_name="hnsw",
    )

    backend = kb._ensure_backend([[1.0, 0.0, 2.0]])
    assert isinstance(backend, DummyHNSWBackend)
    assert backend.dim == 3


def test_ensure_backend_falls_back_to_in_memory_when_backend_is_none():
    kb = KnowledgeBase(
        sources=[],
        embedder=DummyEmbedder(),
        chunker=lambda t, s: [Chunk(text=t, source=s, index=0)],
        retrieval=RetrievalConfig(),
        backend=None,
    )
    kb._backend = None
    backend = kb._ensure_backend([[1.0, 0.0]])
    assert isinstance(backend, InMemoryKnowledgeBackend)
