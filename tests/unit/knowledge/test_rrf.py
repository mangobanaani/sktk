# tests/unit/knowledge/test_rrf.py
from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk, reciprocal_rank_fusion


def _make_scored(source: str, index: int, score: float, method: str = "dense") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(text=f"text {source}:{index}", source=source, index=index),
        score=score,
        retrieval_method=method,
    )


def test_rrf_merges_two_lists():
    list_a = [_make_scored("a", 0, 0.9), _make_scored("b", 0, 0.8)]
    list_b = [_make_scored("b", 0, 0.95), _make_scored("c", 0, 0.7)]
    results = reciprocal_rank_fusion([list_a, list_b], top_k=3)
    assert results[0].chunk.source == "b"
    assert results[0].retrieval_method == "hybrid"


def test_rrf_respects_top_k():
    list_a = [_make_scored("a", i, 1.0 - i * 0.1) for i in range(10)]
    results = reciprocal_rank_fusion([list_a], top_k=3)
    assert len(results) == 3


def test_rrf_empty_lists():
    results = reciprocal_rank_fusion([], top_k=5)
    assert len(results) == 0


def test_rrf_single_list():
    items = [_make_scored("a", 0, 0.9), _make_scored("b", 0, 0.8)]
    results = reciprocal_rank_fusion([items], top_k=5)
    assert len(results) == 2
    assert results[0].score > results[1].score
