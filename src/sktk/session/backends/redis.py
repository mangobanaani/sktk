"""Redis-backed persistent session backends.

Requires the `redis` extra: pip install skat[redis]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from sktk.session.history import ConversationHistory


class RedisHistory(ConversationHistory):
    """Redis-backed persistent conversation history."""

    def __init__(self, url: str = "redis://localhost:6379", session_id: str = "") -> None:
        self._url = url
        self._session_id = session_id
        self._client: Any = None
        self._key = f"skat:history:{session_id}"
        self._count = 0
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> None:
        """Lazily initialize the async Redis client if not yet connected.

        Must be called while holding ``self._lock``.
        """
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise ImportError(
                    "Redis support requires the 'redis' package. "
                    "Install it with: pip install skat[redis]"
                ) from e
            self._client = aioredis.from_url(self._url)
            self._count = await self._client.llen(self._key)

    async def append(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Append a message to the Redis list.

        The lock is held across client initialisation, the rpush, and
        the count increment so that ``__len__`` always reflects the
        actual number of persisted messages.
        """
        message = json.dumps({"role": role, "content": content, "metadata": metadata or {}})
        async with self._lock:
            await self._ensure_client()
            await self._client.rpush(self._key, message)
            self._count += 1

    async def get(self, limit: int | None = None, roles: list[str] | None = None) -> list[dict]:
        """Retrieve messages, optionally filtered by role and limited to the N most recent.

        The lock is held across client initialisation, the lrange call,
        and result processing so that concurrent appends or clears
        cannot interleave and produce an inconsistent snapshot.
        """
        async with self._lock:
            await self._ensure_client()
            raw_messages = await self._client.lrange(self._key, 0, -1)
            messages = [json.loads(m) for m in raw_messages]
            if roles:
                messages = [m for m in messages if m["role"] in roles]
            if limit is not None:
                messages = messages[-limit:]
            return messages

    async def clear(self) -> None:
        """Delete all messages for this session."""
        async with self._lock:
            await self._ensure_client()
            await self._client.delete(self._key)
            self._count = 0

    async def fork(self, session_id: str) -> RedisHistory:
        """Copy all messages into a new RedisHistory under a different session ID.

        The lock is held across client initialisation and the lrange
        read so that concurrent append() or clear() calls cannot
        produce an inconsistent snapshot.  Writing to the forked key
        does not need the lock because the new instance is not yet
        shared.
        """
        async with self._lock:
            await self._ensure_client()
            messages = await self._client.lrange(self._key, 0, -1)
        forked = RedisHistory(url=self._url, session_id=session_id)
        if messages:
            async with forked._lock:
                await forked._ensure_client()
                await forked._client.rpush(forked._key, *messages)
                forked._count = len(messages)
        return forked

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        async with self._lock:
            if self._client is not None:
                await self._client.close()
                self._client = None

    def __len__(self) -> int:
        """Return the cached message count as a best-effort snapshot.

        This is not guaranteed to be consistent under concurrent writes
        because it reads shared state without holding the async lock.
        """
        return self._count
