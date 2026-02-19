"""Text chunking strategies for knowledge base indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Chunk:
    """A chunk of text with source provenance."""

    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


# Type alias for chunker callables
Chunker = Callable[[str, str], list[Chunk]]
ChunkingStrategy = Chunker  # Alias for backward compat


def fixed_size_chunker(max_words: int, overlap_words: int = 0) -> Chunker:
    """Create a chunker that splits text into fixed-size word windows."""
    if overlap_words >= max_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be less than max_words ({max_words})"
        )

    def chunk(text: str, source: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        step = max_words - overlap_words

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(text=chunk_text, source=source, index=idx))
            idx += 1
            if end >= len(words):
                break
            start += step

        return chunks

    return chunk


def sentence_chunker(max_sentences: int) -> Chunker:
    """Create a chunker that groups sentences together."""

    def chunk(text: str, source: str) -> list[Chunk]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return []

        chunks: list[Chunk] = []

        for idx, i in enumerate(range(0, len(sentences), max_sentences)):
            group = sentences[i : i + max_sentences]
            chunk_text = " ".join(group)
            chunks.append(Chunk(text=chunk_text, source=source, index=idx))

        return chunks

    return chunk


def token_count_chunker(
    max_tokens: int, overlap_tokens: int = 0, tokens_per_word: float = 1.3
) -> Chunker:
    """Chunk text by approximate token count using a word-based estimator.

    tokens_per_word lets you align to your target model tokenization (default ~1.3).
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    max_words = int(max_tokens / tokens_per_word)
    overlap_words = int(overlap_tokens / tokens_per_word)
    if max_words <= 0:
        raise ValueError("max_tokens too small for token_count_chunker")
    return fixed_size_chunker(max_words=max_words, overlap_words=overlap_words)
