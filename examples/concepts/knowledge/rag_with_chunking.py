"""RAG with chunking example.

Demonstrates setting up a KnowledgeBase with text chunking
and dense retrieval to answer queries.

Usage:
    python examples/concepts/knowledge/rag_with_chunking.py
"""

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
    """Word-overlap based embedder for demonstration."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.vocab:
            all_words: set[str] = set()
            for t in texts:
                all_words.update(t.lower().split())
            self.vocab = sorted(all_words)
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        words = set(text.lower().split())
        return [1.0 if v in words else 0.0 for v in self.vocab]


async def main() -> None:
    kb = KnowledgeBase(
        sources=[
            TextSource(
                "Python was created by Guido van Rossum. "
                "It was first released in 1991. "
                "Python emphasizes code readability.",
                name="python-facts",
            ),
            TextSource(
                "Rust is a systems programming language. "
                "It focuses on safety and performance. "
                "Rust was created at Mozilla.",
                name="rust-facts",
            ),
        ],
        embedder=SimpleEmbedder(),
        chunker=fixed_size_chunker(max_words=10, overlap_words=2),
        retrieval=RetrievalConfig(mode=RetrievalMode.DENSE, top_k=3),
        backend=InMemoryKnowledgeBackend(),
    )

    await kb.build()
    print(f"Indexed {await kb.chunk_count()} chunks")

    results = await kb.query("Who created Python?")
    print("\nQuery: 'Who created Python?'")
    print(f"Top {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  {i + 1}. [{r.chunk.source}] (score: {r.score:.3f})")
        print(f"     {r.chunk.text}")


if __name__ == "__main__":
    asyncio.run(main())
