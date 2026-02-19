# tests/unit/knowledge/test_memory_backend.py
import pytest

from sktk.knowledge.backends.in_memory import InMemoryKnowledgeBackend
from sktk.knowledge.backends.similarity import cosine_similarity
from sktk.knowledge.chunking import Chunk


@pytest.mark.asyncio
async def test_store_and_search():
    backend = InMemoryKnowledgeBackend()
    chunks = [
        Chunk(text="Python programming", source="a", index=0),
        Chunk(text="Java programming", source="b", index=0),
        Chunk(text="Cooking recipes", source="c", index=0),
    ]
    embeddings = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    await backend.store(chunks, embeddings)
    results = await backend.search([1.0, 0.5, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.text == "Python programming"


@pytest.mark.asyncio
async def test_store_empty():
    backend = InMemoryKnowledgeBackend()
    await backend.store([], [])
    results = await backend.search([1.0], top_k=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_returns_scores():
    backend = InMemoryKnowledgeBackend()
    chunks = [Chunk(text="test", source="a", index=0)]
    await backend.store(chunks, [[1.0, 0.0]])
    results = await backend.search([1.0, 0.0], top_k=1)
    assert results[0].score > 0


@pytest.mark.asyncio
async def test_count():
    backend = InMemoryKnowledgeBackend()
    chunks = [
        Chunk(text="a", source="a", index=0),
        Chunk(text="b", source="b", index=0),
    ]
    await backend.store(chunks, [[1.0], [0.0]])
    assert await backend.count() == 2


@pytest.mark.asyncio
async def test_clear():
    backend = InMemoryKnowledgeBackend()
    chunks = [Chunk(text="x", source="a", index=0)]
    await backend.store(chunks, [[1.0]])
    assert await backend.count() == 1
    await backend.clear()
    assert await backend.count() == 0


@pytest.mark.asyncio
async def test_store_replaces_not_appends():
    backend = InMemoryKnowledgeBackend()
    chunks1 = [Chunk(text="old", source="s", index=0)]
    await backend.store(chunks1, [[0.1]])
    assert await backend.count() == 1

    chunks2 = [Chunk(text="new", source="s", index=0)]
    await backend.store(chunks2, [[0.9]])
    results = await backend.search([0.9], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.text == "new"


@pytest.mark.asyncio
async def test_cosine_similarity_zero_vector():
    backend = InMemoryKnowledgeBackend()
    chunks = [Chunk(text="zero", source="a", index=0)]
    await backend.store(chunks, [[0.0, 0.0]])
    results = await backend.search([1.0, 0.0], top_k=1)
    assert results[0].score == 0.0


@pytest.mark.asyncio
async def test_search_handles_dimension_mismatch():
    backend = InMemoryKnowledgeBackend()
    chunks = [Chunk(text="vec", source="a", index=0)]
    await backend.store(chunks, [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="dimension mismatch"):
        await backend.search([1.0, 0.0], top_k=1)


def test_cosine_similarity_empty_vectors_returns_zero():
    assert cosine_similarity([], []) == 0.0
