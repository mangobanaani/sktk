# tests/unit/knowledge/test_bm25.py
import pytest

from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import BM25Index


@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="Python is a programming language", source="doc1", index=0),
        Chunk(text="Java is also a programming language", source="doc2", index=0),
        Chunk(text="Cooking recipes for pasta and pizza", source="doc3", index=0),
        Chunk(text="Python programming best practices", source="doc4", index=0),
    ]


def test_bm25_index_and_search(sample_chunks):
    idx = BM25Index()
    idx.index(sample_chunks)
    results = idx.search("python programming", top_k=2)
    assert len(results) == 2
    sources = [r.chunk.source for r in results]
    assert "doc1" in sources or "doc4" in sources


def test_bm25_irrelevant_query(sample_chunks):
    idx = BM25Index()
    idx.index(sample_chunks)
    results = idx.search("quantum physics", top_k=5)
    assert len(results) == 0


def test_bm25_empty_index():
    idx = BM25Index()
    idx.index([])
    results = idx.search("anything", top_k=5)
    assert len(results) == 0


def test_bm25_scores_decrease(sample_chunks):
    idx = BM25Index()
    idx.index(sample_chunks)
    results = idx.search("python programming", top_k=10)
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_bm25_retrieval_method(sample_chunks):
    idx = BM25Index()
    idx.index(sample_chunks)
    results = idx.search("python", top_k=1)
    assert results[0].retrieval_method == "sparse"


def test_bm25_tokenization_strips_punctuation():
    idx = BM25Index()
    chunks = [Chunk(text="Hello, world!", source="d", index=0)]
    idx.index(chunks)
    results = idx.search("hello world?", top_k=1)
    assert results
