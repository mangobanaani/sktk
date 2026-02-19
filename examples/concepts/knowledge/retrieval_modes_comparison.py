"""Compare dense, sparse, and hybrid retrieval modes.

Builds the same knowledge corpus three times with different retrieval modes so
you can see how ranking behavior changes.

Usage:
    python examples/concepts/knowledge/retrieval_modes_comparison.py
"""

from __future__ import annotations

import asyncio

from sktk.knowledge import (
    InMemoryKnowledgeBackend,
    KnowledgeBase,
    RetrievalConfig,
    RetrievalMode,
    TextSource,
    fixed_size_chunker,
)


class SimpleEmbedder:
    """Tiny word-overlap embedder for deterministic demo output."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.vocab:
            words: set[str] = set()
            for text in texts:
                words.update(text.lower().split())
            self.vocab = sorted(words)
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        tokens = set(text.lower().split())
        return [1.0 if token in tokens else 0.0 for token in self.vocab]


SOURCES = [
    TextSource(
        "Python is an interpreted language created by Guido van Rossum in 1991. "
        "Python emphasizes readability and developer productivity.",
        name="python-overview",
    ),
    TextSource(
        "Rust is a compiled systems language focused on memory safety and performance. "
        "Rust ownership rules prevent data races.",
        name="rust-overview",
    ),
    TextSource(
        "FastAPI is a Python web framework known for type hints and async support.",
        name="fastapi-overview",
    ),
]


async def build_kb(mode: RetrievalMode) -> KnowledgeBase:
    kb = KnowledgeBase(
        sources=SOURCES,
        embedder=SimpleEmbedder(),
        chunker=fixed_size_chunker(max_words=14, overlap_words=2),
        retrieval=RetrievalConfig(mode=mode, top_k=3),
        backend=InMemoryKnowledgeBackend(),
    )
    await kb.build()
    return kb


async def main() -> None:
    query = "Which language focuses on memory safety?"

    for mode in [RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID]:
        kb = await build_kb(mode)
        results = await kb.query(query)

        print(f"\n=== {mode.value.upper()} ===")
        for idx, result in enumerate(results, start=1):
            print(
                f"  {idx}. source={result.chunk.source:<16} "
                f"method={result.retrieval_method:<6} score={result.score:.4f}"
            )
            print(f"     {result.chunk.text}")


if __name__ == "__main__":
    asyncio.run(main())
