# tests/unit/session/test_redis_history.py
"""Tests for RedisHistory with mocked redis.asyncio."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from sktk.session.backends.redis import RedisHistory


class FakeRedisClient:
    """In-memory fake for redis.asyncio client."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {}

    async def rpush(self, key: str, *values: str) -> None:
        if key not in self._data:
            self._data[key] = []
        self._data[key].extend(values)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self._data.get(key, [])

    async def llen(self, key: str) -> int:
        return len(self._data.get(key, []))

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_redis():
    """Inject a FakeRedisClient directly into RedisHistory, bypassing import."""
    client = FakeRedisClient()
    yield client


def _make_history(client: FakeRedisClient, session_id: str) -> RedisHistory:
    """Create a RedisHistory that uses the fake client directly."""
    h = RedisHistory(url="redis://localhost", session_id=session_id)
    # Bypass _get_client import by injecting client directly
    h._client = client
    return h


@pytest.mark.asyncio
async def test_redis_history_append_and_get(fake_redis):
    h = _make_history(fake_redis, "s1")
    await h.append("user", "hello")
    await h.append("assistant", "hi there")
    messages = await h.get()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["content"] == "hi there"


@pytest.mark.asyncio
async def test_redis_history_len(fake_redis):
    h = _make_history(fake_redis, "s2")
    assert len(h) == 0
    await h.append("user", "a")
    assert len(h) == 1
    await h.append("user", "b")
    assert len(h) == 2


@pytest.mark.asyncio
async def test_redis_history_get_with_roles(fake_redis):
    h = _make_history(fake_redis, "s3")
    await h.append("user", "q1")
    await h.append("assistant", "a1")
    await h.append("user", "q2")
    messages = await h.get(roles=["user"])
    assert len(messages) == 2
    assert all(m["role"] == "user" for m in messages)


@pytest.mark.asyncio
async def test_redis_history_get_with_limit(fake_redis):
    h = _make_history(fake_redis, "s4")
    await h.append("user", "a")
    await h.append("user", "b")
    await h.append("user", "c")
    messages = await h.get(limit=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "b"
    assert messages[1]["content"] == "c"


@pytest.mark.asyncio
async def test_redis_history_clear(fake_redis):
    h = _make_history(fake_redis, "s5")
    await h.append("user", "x")
    assert len(h) == 1
    await h.clear()
    assert len(h) == 0
    messages = await h.get()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_redis_history_fork(fake_redis):
    h = _make_history(fake_redis, "s6")
    await h.append("user", "msg1")
    await h.append("assistant", "msg2")

    # For fork, _get_client is called on the new instance.
    forked_client = FakeRedisClient()
    forked_client._data = fake_redis._data  # share storage

    mock_aioredis = MagicMock()
    mock_aioredis.from_url = MagicMock(return_value=forked_client)
    # import redis.asyncio resolves as sys.modules['redis'].asyncio
    mock_redis_mod = MagicMock()
    mock_redis_mod.asyncio = mock_aioredis

    with patch.dict(sys.modules, {"redis": mock_redis_mod, "redis.asyncio": mock_aioredis}):
        forked = await h.fork("s6-fork")
    assert len(forked) == 2
    fork_msgs = await forked.get()
    assert len(fork_msgs) == 2


@pytest.mark.asyncio
async def test_redis_history_fork_empty(fake_redis):
    h = _make_history(fake_redis, "s7")
    forked = await h.fork("s7-fork")
    assert len(forked) == 0


@pytest.mark.asyncio
async def test_redis_history_close(fake_redis):
    h = _make_history(fake_redis, "s8")
    await h.append("user", "x")
    await h.close()
    assert h._client is None


@pytest.mark.asyncio
async def test_redis_history_close_when_not_connected():
    h = RedisHistory(url="redis://localhost", session_id="s9")
    await h.close()  # should not raise


@pytest.mark.asyncio
async def test_redis_history_ensure_client_import():
    """Test _ensure_client with mocked redis.asyncio import."""
    client = FakeRedisClient()
    mock_aioredis = MagicMock()
    mock_aioredis.from_url = MagicMock(return_value=client)
    mock_redis_mod = MagicMock()
    mock_redis_mod.asyncio = mock_aioredis

    h = RedisHistory(url="redis://localhost", session_id="s10")
    with patch.dict(sys.modules, {"redis": mock_redis_mod, "redis.asyncio": mock_aioredis}):
        async with h._lock:
            await h._ensure_client()
    assert h._client is client
    assert h._count == 0


@pytest.mark.asyncio
async def test_redis_history_import_error():
    """Test _ensure_client raises ImportError when redis not installed."""
    h = RedisHistory(url="redis://localhost", session_id="s11")

    with (
        patch.dict(sys.modules, {"redis": None, "redis.asyncio": None}),
        pytest.raises(ImportError, match="redis"),
    ):
        async with h._lock:
            await h._ensure_client()
