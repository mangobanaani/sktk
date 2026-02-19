# tests/unit/knowledge/test_retrieval_config.py
from sktk.knowledge.retrieval import RetrievalConfig, RetrievalMode


def test_retrieval_mode_values():
    assert RetrievalMode.DENSE.value == "dense"
    assert RetrievalMode.SPARSE.value == "sparse"
    assert RetrievalMode.HYBRID.value == "hybrid"


def test_retrieval_config_defaults():
    config = RetrievalConfig()
    assert config.mode == RetrievalMode.DENSE
    assert config.top_k == 5
    assert config.reranker is None


def test_retrieval_config_custom():
    config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
        top_k=10,
    )
    assert config.mode == RetrievalMode.HYBRID
    assert config.top_k == 10


def test_retrieval_config_ttl():
    cfg = RetrievalConfig(ttl_seconds=3600)
    assert cfg.ttl_seconds == 3600
