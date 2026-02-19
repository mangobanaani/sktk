"""In-memory implementations of session backends."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
from typing import Any, AsyncIterator, TypeVar

from pydantic import BaseModel, ValidationError

from sktk.core.errors import BlackboardTypeError
from sktk.session.blackboard import Blackboard
from sktk.session.history import ConversationHistory

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class InMemoryHistory(ConversationHistory):
    """In-memory conversation history backed by a plain list."""

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._lock = asyncio.Lock()

    async def append(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Append a message to the in-memory list."""
        async with self._lock:
            self._messages.append({"role": role, "content": content, "metadata": metadata or {}})

    async def get(self, limit: int | None = None, roles: list[str] | None = None) -> list[dict]:
        """Return deep-copied messages, optionally filtered and limited."""
        async with self._lock:
            messages = self._messages
            if roles:
                messages = [m for m in messages if m["role"] in roles]
            if limit is not None:
                messages = messages[-limit:]
            return [copy.deepcopy(m) for m in messages]

    async def clear(self) -> None:
        """Remove all messages."""
        async with self._lock:
            self._messages.clear()

    async def fork(self, session_id: str) -> InMemoryHistory:
        """Create an independent deep copy of this history.

        The *session_id* parameter is accepted for protocol compatibility
        but is unused by the in-memory backend.
        """
        async with self._lock:
            forked = InMemoryHistory()
            forked._messages = copy.deepcopy(self._messages)
            return forked

    async def close(self) -> None:
        """No-op close for interface compatibility."""
        pass

    def __len__(self) -> int:
        """Return the number of messages as a best-effort snapshot.

        This is not guaranteed to be consistent under concurrent writes
        because it reads shared state without holding the async lock.
        """
        return len(self._messages)


class InMemoryBlackboard(Blackboard):
    """In-memory blackboard storing serialized Pydantic models as JSON strings."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._type_names: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._watchers: dict[str, list[asyncio.Queue]] = {}

    async def set(self, key: str, value: BaseModel) -> None:
        """Store a value and notify any watchers on that key."""
        notifications: list[asyncio.Queue] = []
        async with self._lock:
            self._data[key] = value.model_dump_json()
            self._type_names[key] = type(value).__name__
            if key in self._watchers:
                notifications = list(self._watchers[key])
        for queue in notifications:
            try:
                queue.put_nowait(value.model_copy(deep=True))
            except asyncio.QueueFull:
                logger.warning("Watcher queue full for key %r; skipping notification", key)

    async def get(self, key: str, model: type[T]) -> T | None:
        """Retrieve and validate a value, raising BlackboardTypeError on mismatch."""
        async with self._lock:
            raw = self._data.get(key)
            if raw is None:
                return None
            try:
                return model.model_validate_json(raw)
            except ValidationError as e:
                raise BlackboardTypeError(
                    key=key, expected=model.__name__, got=self._type_names.get(key, "unknown")
                ) from e

    async def get_all(self, prefix: str) -> dict[str, Any]:
        """Return all entries whose keys start with prefix as parsed dicts."""
        async with self._lock:
            return {k: json.loads(v) for k, v in self._data.items() if k.startswith(prefix)}

    async def delete(self, key: str) -> bool:
        """Delete a key and return True if it existed."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
                self._type_names.pop(key, None)
                return True
            return False

    async def watch(self, key: str) -> AsyncIterator[BaseModel]:
        """Yield new values for a key as they are set, using an asyncio.Queue per watcher."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            if key not in self._watchers:
                self._watchers[key] = []
            self._watchers[key].append(queue)
        try:
            while True:
                value = await queue.get()
                yield value
        finally:
            async with self._lock:
                if key in self._watchers:
                    with contextlib.suppress(ValueError):
                        self._watchers[key].remove(queue)

    async def close(self) -> None:
        """No-op close for interface compatibility."""
        pass

    async def keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        async with self._lock:
            return [k for k in self._data if k.startswith(prefix)]
