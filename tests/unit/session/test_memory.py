# tests/unit/session/test_memory.py
"""Tests for SemanticMemory and MemoryGroundingFilter."""

from unittest.mock import AsyncMock

import pytest

from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Modify
from sktk.knowledge.chunking import Chunk
from sktk.knowledge.retrieval import ScoredChunk
from sktk.session.memory import MemoryEntry, MemoryGroundingFilter, SemanticMemory


def _make_mock_kb():
    """Create a mock KnowledgeBase with async methods."""
    kb = AsyncMock()
    kb.add_source = AsyncMock()
    kb.query = AsyncMock(return_value=[])
    return kb


def _make_scored_chunk(text: str, source: str, score: float) -> ScoredChunk:
    """Helper to build a ScoredChunk for test assertions."""
    return ScoredChunk(
        chunk=Chunk(text=text, source=source, index=0),
        score=score,
    )


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


class TestSemanticMemoryRemember:
    @pytest.mark.asyncio
    async def test_remember_stores_entry_and_delegates_to_kb(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("pref", "formal tone")

        assert "pref" in await mem.list_keys()
        kb.add_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_creates_memory_entry_with_content(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("style", "concise answers")

        entry = mem._keys["style"]
        assert isinstance(entry, MemoryEntry)
        assert entry.key == "style"
        assert entry.content == "concise answers"
        assert entry.timestamp > 0

    @pytest.mark.asyncio
    async def test_remember_passes_text_source_to_kb(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("lang", "uses Python")

        source_arg = kb.add_source.call_args[0][0]
        assert source_arg.name == "memory:lang:v1"
        loaded = await source_arg.load()
        assert loaded == "uses Python"


class TestSemanticMemoryOverwrite:
    @pytest.mark.asyncio
    async def test_overwrite_same_key_updates_entry(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("key", "first value")
        await mem.remember("key", "second value")

        assert mem._keys["key"].content == "second value"
        assert await mem.list_keys() == ["key"]
        assert kb.add_source.call_count == 2

    @pytest.mark.asyncio
    async def test_overwrite_updates_timestamp(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("key", "v1")
        ts1 = mem._keys["key"].timestamp

        await mem.remember("key", "v2")
        ts2 = mem._keys["key"].timestamp

        assert ts2 >= ts1


class TestSemanticMemoryRecall:
    @pytest.mark.asyncio
    async def test_recall_delegates_to_kb_query(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.recall("what tone?")

        kb.query.assert_called_once_with("what tone?")

    @pytest.mark.asyncio
    async def test_recall_formats_results(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("formal tone", "memory:pref", 0.95),
            _make_scored_chunk("concise style", "memory:style", 0.80),
        ]
        mem = SemanticMemory(knowledge_base=kb)

        results = await mem.recall("how to respond?")

        assert len(results) == 2
        assert results[0] == {"text": "formal tone", "score": 0.95, "source": "memory:pref"}
        assert results[1] == {"text": "concise style", "score": 0.80, "source": "memory:style"}

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("a", "s1", 0.9),
            _make_scored_chunk("b", "s2", 0.8),
            _make_scored_chunk("c", "s3", 0.7),
        ]
        mem = SemanticMemory(knowledge_base=kb)

        results = await mem.recall("query", top_k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_recall_empty_results(self):
        kb = _make_mock_kb()
        kb.query.return_value = []
        mem = SemanticMemory(knowledge_base=kb)

        results = await mem.recall("no match")

        assert results == []


class TestSemanticMemoryForget:
    @pytest.mark.asyncio
    async def test_forget_existing_key_returns_true(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("key", "value")
        result = await mem.forget("key")

        assert result is True
        assert "key" not in await mem.list_keys()

    @pytest.mark.asyncio
    async def test_forget_nonexistent_key_returns_false(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        result = await mem.forget("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_forget_removes_from_keys_dict(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("a", "1")
        await mem.remember("b", "2")
        await mem.forget("a")

        assert await mem.list_keys() == ["b"]


class TestSemanticMemoryListKeys:
    @pytest.mark.asyncio
    async def test_list_keys_empty(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        assert await mem.list_keys() == []

    @pytest.mark.asyncio
    async def test_list_keys_multiple(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)

        await mem.remember("a", "1")
        await mem.remember("b", "2")
        await mem.remember("c", "3")

        keys = await mem.list_keys()
        assert set(keys) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# MemoryGroundingFilter
# ---------------------------------------------------------------------------


def _make_filter_context(content: str) -> FilterContext:
    return FilterContext(content=content, stage="input", agent_name="test-agent")


class TestMemoryGroundingFilterOnInput:
    @pytest.mark.asyncio
    async def test_no_relevant_memories_returns_allow(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem)

        result = await grounding.on_input(_make_filter_context("hello"))

        assert isinstance(result, Allow)

    @pytest.mark.asyncio
    async def test_relevant_memories_returns_modify_with_context(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("user likes formal tone", "memory:pref", 0.9),
        ]
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem)

        result = await grounding.on_input(_make_filter_context("how should I respond?"))

        assert isinstance(result, Modify)
        assert "user likes formal tone" in result.content
        assert "how should I respond?" in result.content

    @pytest.mark.asyncio
    async def test_modify_prepends_prefix(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("context A", "s1", 0.8),
        ]
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem, prefix="[Memory]\n")

        result = await grounding.on_input(_make_filter_context("query"))

        assert isinstance(result, Modify)
        assert result.content.startswith("[Memory]\n")

    @pytest.mark.asyncio
    async def test_modify_includes_all_relevant_memories(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("fact A", "s1", 0.9),
            _make_scored_chunk("fact B", "s2", 0.8),
        ]
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem)

        result = await grounding.on_input(_make_filter_context("query"))

        assert isinstance(result, Modify)
        assert "- fact A" in result.content
        assert "- fact B" in result.content

    @pytest.mark.asyncio
    async def test_min_score_filters_low_scoring_memories(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("high relevance", "s1", 0.9),
            _make_scored_chunk("low relevance", "s2", 0.3),
        ]
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem, min_score=0.5)

        result = await grounding.on_input(_make_filter_context("query"))

        assert isinstance(result, Modify)
        assert "high relevance" in result.content
        assert "low relevance" not in result.content

    @pytest.mark.asyncio
    async def test_min_score_filters_all_returns_allow(self):
        kb = _make_mock_kb()
        kb.query.return_value = [
            _make_scored_chunk("low", "s1", 0.2),
        ]
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem, min_score=0.5)

        result = await grounding.on_input(_make_filter_context("query"))

        assert isinstance(result, Allow)


class TestMemoryGroundingFilterOnOutput:
    @pytest.mark.asyncio
    async def test_on_output_returns_allow(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem)

        ctx = FilterContext(content="output text", stage="output", agent_name="test")
        result = await grounding.on_output(ctx)

        assert isinstance(result, Allow)


class TestMemoryGroundingFilterOnFunctionCall:
    @pytest.mark.asyncio
    async def test_on_function_call_returns_allow(self):
        kb = _make_mock_kb()
        mem = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=mem)

        ctx = FilterContext(content="fn call", stage="function_call", agent_name="test")
        result = await grounding.on_function_call(ctx)

        assert isinstance(result, Allow)
