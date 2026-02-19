# tests/unit/knowledge/test_chunking.py
import pytest

from sktk.knowledge.chunking import (
    Chunk,
    fixed_size_chunker,
    sentence_chunker,
    token_count_chunker,
)


def test_chunk_creation():
    c = Chunk(text="hello world", source="doc.txt", index=0, metadata={"page": 1})
    assert c.text == "hello world"
    assert c.source == "doc.txt"
    assert c.index == 0
    assert c.metadata == {"page": 1}


def test_fixed_size_chunker_basic():
    text = "word " * 100  # 100 words
    chunker = fixed_size_chunker(max_words=20, overlap_words=5)
    chunks = chunker(text, source="test.txt")
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "test.txt" for c in chunks)


def test_fixed_size_chunker_small_text():
    text = "short text"
    chunker = fixed_size_chunker(max_words=20, overlap_words=5)
    chunks = chunker(text, source="test.txt")
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "short text"


def test_fixed_size_chunker_overlap():
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunker = fixed_size_chunker(max_words=10, overlap_words=3)
    chunks = chunker(text, source="test.txt")
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-3:] == second_words[:3]


def test_sentence_chunker_basic():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunker = sentence_chunker(max_sentences=2)
    chunks = chunker(text, source="test.txt")
    assert len(chunks) == 3
    assert "First" in chunks[0].text
    assert "Third" in chunks[1].text


def test_sentence_chunker_single_sentence():
    text = "Just one sentence."
    chunker = sentence_chunker(max_sentences=3)
    chunks = chunker(text, source="test.txt")
    assert len(chunks) == 1


def test_chunks_have_sequential_indices():
    text = "word " * 100
    chunker = fixed_size_chunker(max_words=20, overlap_words=0)
    chunks = chunker(text, source="test.txt")
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_fixed_size_chunker_overlap_exceeds_max():
    import pytest

    with pytest.raises(ValueError, match="overlap_words"):
        fixed_size_chunker(max_words=5, overlap_words=5)


def test_fixed_size_chunker_empty_text():
    chunker = fixed_size_chunker(max_words=10)
    chunks = chunker("", source="empty.txt")
    assert chunks == []


def test_sentence_chunker_empty_text():
    chunker = sentence_chunker(max_sentences=2)
    chunks = chunker("", source="empty.txt")
    assert chunks == []


def test_token_count_chunker():
    chunker = token_count_chunker(max_tokens=10, overlap_tokens=5, tokens_per_word=1.0)
    text = "one two three four five six seven eight nine ten eleven"
    chunks = chunker(text, "s")
    assert len(chunks) == 2
    assert chunks[0].text.startswith("one two three")
    assert chunks[1].text.startswith("six seven eight")


def test_token_count_chunker_rejects_overlap_equal_to_max():
    with pytest.raises(ValueError, match="overlap_tokens"):
        token_count_chunker(max_tokens=5, overlap_tokens=5, tokens_per_word=1.0)


def test_token_count_chunker_rejects_tiny_max_tokens():
    with pytest.raises(ValueError, match="max_tokens too small"):
        token_count_chunker(max_tokens=1, overlap_tokens=0, tokens_per_word=10.0)
