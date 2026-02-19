import asyncio

import pytest
from pydantic import BaseModel

from sktk.core.errors import BlackboardTypeError
from sktk.session.backends.memory import InMemoryBlackboard


class TaskResult(BaseModel):
    summary: str
    score: float


class OtherModel(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_set_and_get():
    bb = InMemoryBlackboard()
    result = TaskResult(summary="good", score=0.95)
    await bb.set("result", result)
    got = await bb.get("result", TaskResult)
    assert got is not None
    assert got.summary == "good"
    assert got.score == 0.95


@pytest.mark.asyncio
async def test_get_missing_key():
    bb = InMemoryBlackboard()
    assert await bb.get("missing", TaskResult) is None


@pytest.mark.asyncio
async def test_get_wrong_type():
    bb = InMemoryBlackboard()
    await bb.set("result", TaskResult(summary="good", score=0.95))
    with pytest.raises(BlackboardTypeError):
        await bb.get("result", OtherModel)


@pytest.mark.asyncio
async def test_get_all_with_prefix():
    bb = InMemoryBlackboard()
    await bb.set("agent.a.result", TaskResult(summary="a", score=0.9))
    await bb.set("agent.b.result", TaskResult(summary="b", score=0.8))
    await bb.set("other.key", TaskResult(summary="c", score=0.7))
    results = await bb.get_all("agent.")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete():
    bb = InMemoryBlackboard()
    await bb.set("key", TaskResult(summary="x", score=1.0))
    assert await bb.delete("key") is True
    assert await bb.get("key", TaskResult) is None
    assert await bb.delete("key") is False


@pytest.mark.asyncio
async def test_keys():
    bb = InMemoryBlackboard()
    await bb.set("a.1", TaskResult(summary="x", score=1.0))
    await bb.set("a.2", TaskResult(summary="y", score=2.0))
    await bb.set("b.1", TaskResult(summary="z", score=3.0))
    assert sorted(await bb.keys()) == ["a.1", "a.2", "b.1"]
    assert sorted(await bb.keys("a.")) == ["a.1", "a.2"]


@pytest.mark.asyncio
async def test_watch():
    bb = InMemoryBlackboard()
    received = []

    async def watcher():
        async for value in bb.watch("key"):
            received.append(value)
            if len(received) >= 2:
                break

    task = asyncio.create_task(watcher())
    await asyncio.sleep(0.01)
    await bb.set("key", TaskResult(summary="first", score=1.0))
    await asyncio.sleep(0.01)
    await bb.set("key", TaskResult(summary="second", score=2.0))
    await asyncio.sleep(0.01)
    await task
    assert len(received) == 2
