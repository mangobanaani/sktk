"""Semantic memory -- cross-session memory backed by knowledge base.

Provides remember/recall/forget operations with embedding-based retrieval,
plus a MemoryGroundingFilter that auto-injects relevant memories into context.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, FilterResult, Modify

if TYPE_CHECKING:
    from sktk.knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory with key, content, and timestamp."""

    key: str
    content: str
    timestamp: float


class SemanticMemory:
    """Cross-session semantic memory backed by a KnowledgeBase.

    Usage:
        from sktk.knowledge.knowledge_base import KnowledgeBase
        memory = SemanticMemory(knowledge_base=kb)
        await memory.remember("user_preference", "User prefers formal tone")
        results = await memory.recall("How should I address the user?", top_k=3)
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base
        self._keys: dict[str, MemoryEntry] = {}
        self._versions: dict[str, int] = {}
        self._stale_sources: set[str] = set()
        self._lock = asyncio.Lock()

    def _source_name(self, key: str, version: int) -> str:
        """Return a versioned source name for a memory key."""
        return f"memory:{key}:v{version}"

    async def remember(self, key: str, content: str) -> None:
        """Store a memory with a key. Overwrites if key exists.

        On overwrite the old source name is marked stale so that
        ``recall()`` filters out results from superseded versions.
        """
        async with self._lock:
            old_version = self._versions.get(key, 0)
            version = old_version + 1
            self._versions[key] = version
            entry = MemoryEntry(key=key, content=content, timestamp=time.time())
            self._keys[key] = entry
            source_name = self._source_name(key, version)
            # Mark previous version as stale
            if old_version > 0:
                self._stale_sources.add(self._source_name(key, old_version))
        # add_source outside lock -- KB has its own internal lock
        from sktk.knowledge.knowledge_base import TextSource

        await self._kb.add_source(TextSource(content=content, name=source_name))
        logger.debug("Remembered: %s (version %d)", key, version)

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        """Retrieve relevant memories by semantic similarity.

        Filters out results from superseded (stale) memory versions so
        that only the latest version of each key is returned.

        The *top_k* parameter controls how many results are returned from
        those retrieved by the knowledge base. Note: the knowledge base's
        own ``RetrievalConfig.top_k`` determines how many candidates are
        fetched from the index; this parameter only slices the final list.
        To retrieve more candidates, configure the knowledge base with a
        higher ``top_k``.
        """
        results = await self._kb.query(query)
        async with self._lock:
            stale = set(self._stale_sources)
        filtered = []
        for r in results:
            source = r.chunk.source
            # Skip results from superseded memory versions
            if source in stale:
                continue
            filtered.append(
                {
                    "text": r.chunk.text,
                    "score": r.score,
                    "source": source,
                }
            )
            if len(filtered) >= top_k:
                break
        return filtered

    async def forget(self, key: str) -> bool:
        """Remove a memory key from tracking.

        Note: The content remains in the knowledge base index until it is
        rebuilt. Use this to stop tracking the key; a full rebuild of the
        knowledge base is needed to remove the content from search results.
        Meanwhile, ``recall()`` will filter out results from forgotten keys.

        Returns True if the key was found and removed from tracking.
        """
        async with self._lock:
            if key in self._keys:
                del self._keys[key]
                # Mark all versions of this key as stale so recall() filters them out
                version = self._versions.get(key, 0)
                for v in range(1, version + 1):
                    self._stale_sources.add(self._source_name(key, v))
                logger.debug("Forgot: %s", key)
                return True
            return False

    async def list_keys(self) -> list[str]:
        """List all stored memory keys."""
        async with self._lock:
            return list(self._keys.keys())


class MemoryGroundingFilter:
    """Filter that auto-injects relevant memories into the input context.

    When attached to an agent, this filter queries semantic memory for
    relevant context and prepends it to the user's input.

    Usage:
        memory = SemanticMemory(knowledge_base=kb)
        grounding = MemoryGroundingFilter(memory=memory, top_k=3)
        agent = SKTKAgent(name="a", filters=[grounding, ...])
    """

    def __init__(
        self,
        memory: SemanticMemory,
        top_k: int = 3,
        min_score: float = 0.0,
        prefix: str = "[Relevant context]\n",
    ) -> None:
        self._memory = memory
        self._top_k = top_k
        self._min_score = min_score
        self._prefix = prefix

    async def on_input(self, context: FilterContext) -> FilterResult:
        """Retrieve relevant memories and prepend to input."""
        results = await self._memory.recall(context.content, top_k=self._top_k)
        relevant = [r for r in results if r["score"] >= self._min_score]
        if not relevant:
            return Allow()
        memories_text = "\n".join(f"- {r['text']}" for r in relevant)
        augmented = f"{self._prefix}{memories_text}\n\n{context.content}"
        return Modify(content=augmented)

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return Allow()
