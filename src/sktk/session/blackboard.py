"""Abstract blackboard (shared agent memory) interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Blackboard(ABC):
    """Interface for typed key/value shared state across agents."""

    @abstractmethod
    async def set(self, key: str, value: BaseModel) -> None:
        """Store a Pydantic model under the given key."""
        ...

    @abstractmethod
    async def get(self, key: str, model: type[T]) -> T | None:
        """Retrieve and validate a value by key, returning None if absent."""
        ...

    @abstractmethod
    async def get_all(self, prefix: str) -> dict[str, Any]:
        """Return all key-value pairs whose keys start with prefix."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key and return True if it existed."""
        ...

    @abstractmethod
    async def watch(self, key: str) -> AsyncIterator[BaseModel]:
        """Yield new values for a key as they are set."""
        yield  # type: ignore[misc]

    @abstractmethod
    async def keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        ...
