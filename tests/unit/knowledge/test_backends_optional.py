import importlib
import sys
from types import ModuleType

import pytest

from sktk.knowledge.backends.ann import ANNBackend
from sktk.knowledge.backends.similarity import cosine_similarity
from sktk.knowledge.chunking import Chunk


def _load_optional_backend_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    dependency_name: str,
    dependency_module: ModuleType,
) -> ModuleType:
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, dependency_name, dependency_module)
    return importlib.import_module(module_name)


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(text=text, source="test", index=index)


class _FakeFaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.reset_calls = 0
        self.add_calls = 0
        self.search_calls = 0
        self.search_distances = [[0.0]]
        self.search_indices = [[0]]

    def reset(self) -> None:
        self.reset_calls += 1

    def add(self, vecs) -> None:
        self.add_calls += 1
        self.last_added = vecs

    def search(self, query, top_k: int):
        self.search_calls += 1
        self.last_query = query
        self.last_top_k = top_k
        return self.search_distances, self.search_indices


class _FakeFaissModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("faiss")
        self.created_indices: list[_FakeFaissIndex] = []

    def IndexFlatL2(self, dim: int) -> _FakeFaissIndex:
        index = _FakeFaissIndex(dim)
        self.created_indices.append(index)
        return index


class _FakeHNSWIndex:
    def __init__(self, space: str, dim: int) -> None:
        self.space = space
        self.dim = dim
        self.M = 16
        self.ef = None
        self.init_calls: list[tuple[int, int, int]] = []
        self.add_calls = 0
        self.knn_calls = 0
        self.labels = [[0]]
        self.distances = [[0.0]]

    def init_index(self, max_elements: int, ef_construction: int, M: int) -> None:
        self.M = M
        self.init_calls.append((max_elements, ef_construction, M))

    def set_ef(self, ef_search: int) -> None:
        self.ef = ef_search

    def add_items(self, vecs, labels) -> None:
        self.add_calls += 1
        self.last_added = vecs
        self.last_labels = labels

    def knn_query(self, query, k: int):
        self.knn_calls += 1
        self.last_query = query
        self.last_k = k
        return self.labels, self.distances


class _FakeHNSWModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("hnswlib")
        self.created_indices: list[_FakeHNSWIndex] = []

    def Index(self, space: str, dim: int) -> _FakeHNSWIndex:
        index = _FakeHNSWIndex(space, dim)
        self.created_indices.append(index)
        return index


@pytest.mark.asyncio
async def test_ann_backend_empty_search_returns_empty():
    backend = ANNBackend()
    await backend.store([], [])
    assert await backend.search([0.1], top_k=1) == []


@pytest.mark.asyncio
async def test_ann_backend_store_search_count_and_clear():
    backend = ANNBackend()
    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    await backend.store(chunks, [[1.0, 0.0], [0.0, 1.0]])

    assert await backend.count() == 2
    results = await backend.search([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.text == "alpha"
    assert results[0].retrieval_method == "ann"

    await backend.clear()
    assert await backend.count() == 0
    assert await backend.search([1.0, 0.0], top_k=1) == []


def test_ann_backend_cosine_similarity_edge_cases():
    # Dimension mismatch raises ValueError
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([], [1.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([3.0, 4.0, 999.0], [3.0, 4.0])
    # Zero-norm vectors return 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


@pytest.mark.asyncio
async def test_faiss_backend_store_search_clear_and_skip_missing_indices(monkeypatch):
    fake_faiss = _FakeFaissModule()
    backend_module = _load_optional_backend_module(
        monkeypatch,
        "sktk.knowledge.backends.faiss_backend",
        "faiss",
        fake_faiss,
    )
    backend = backend_module.FaissBackend(dim=2)
    index = fake_faiss.created_indices[0]

    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    await backend.store(chunks, [[1.0, 0.0], [0.0, 1.0]])

    index.search_distances = [[0.25, 9.0, 0.75]]
    index.search_indices = [[0, -1, 1]]
    results = await backend.search([1.0, 0.0], top_k=3)

    assert [r.chunk.text for r in results] == ["alpha", "beta"]
    # Scores use 1/(1+dist): dist=0.25 -> 0.8, dist=0.75 -> ~0.5714
    assert [r.score for r in results] == pytest.approx([1.0 / 1.25, 1.0 / 1.75])
    assert all(r.retrieval_method == "faiss" for r in results)
    assert await backend.count() == 2
    assert index.reset_calls == 1
    assert index.add_calls == 1
    assert index.search_calls == 1

    await backend.clear()
    assert await backend.count() == 0
    assert index.reset_calls == 2
    assert await backend.search([1.0, 0.0], top_k=3) == []
    assert index.search_calls == 1


@pytest.mark.asyncio
async def test_faiss_backend_store_dimension_mismatch_raises(monkeypatch):
    fake_faiss = _FakeFaissModule()
    backend_module = _load_optional_backend_module(
        monkeypatch,
        "sktk.knowledge.backends.faiss_backend",
        "faiss",
        fake_faiss,
    )
    backend = backend_module.FaissBackend(dim=3)
    index = fake_faiss.created_indices[0]

    with pytest.raises(ValueError, match="Embedding dimension mismatch for FAISS backend"):
        await backend.store([_chunk("alpha", 0)], [[1.0, 2.0]])

    assert index.reset_calls == 0
    assert index.add_calls == 0


@pytest.mark.asyncio
async def test_hnsw_backend_constructor_store_search_and_clear(monkeypatch):
    fake_hnsw = _FakeHNSWModule()
    backend_module = _load_optional_backend_module(
        monkeypatch,
        "sktk.knowledge.backends.hnsw_backend",
        "hnswlib",
        fake_hnsw,
    )
    backend = backend_module.HNSWBackend(dim=2, space="cosine", ef_search=77, M=7)
    index = fake_hnsw.created_indices[0]

    assert index.init_calls[0] == (1, 100, 7)
    assert index.ef == 77
    assert await backend.search([1.0, 0.0], top_k=2) == []

    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    await backend.store(chunks, [[1.0, 0.0], [0.0, 1.0]])
    assert index.init_calls[1] == (2, 100, 7)
    assert index.add_calls == 1
    assert index.last_labels == [0, 1]

    index.labels = [[1, 0]]
    index.distances = [[0.2, 0.8]]
    results = await backend.search([0.1, 0.9], top_k=2)

    assert [r.chunk.text for r in results] == ["beta", "alpha"]
    assert [r.score for r in results] == pytest.approx([0.8, 0.2])
    assert all(r.retrieval_method == "hnsw" for r in results)
    assert index.knn_calls == 1
    assert await backend.count() == 2

    await backend.clear()
    assert await backend.count() == 0
    assert index.init_calls[-1] == (1, 100, 7)
