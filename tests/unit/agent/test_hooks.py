# tests/unit/agent/test_hooks.py
import pytest

from sktk.agent.hooks import LifecycleHooks


@pytest.mark.asyncio
async def test_fire_start():
    log = []

    async def on_start(agent_name, input_text):
        log.append(("start", agent_name, input_text))

    hooks = LifecycleHooks(on_start=[on_start])
    await hooks.fire_start("agent1", "hello")
    assert log == [("start", "agent1", "hello")]


@pytest.mark.asyncio
async def test_fire_complete():
    log = []

    async def on_complete(agent_name, input_text, output):
        log.append(("complete", agent_name, output))

    hooks = LifecycleHooks(on_complete=[on_complete])
    await hooks.fire_complete("agent1", "hello", "world")
    assert log == [("complete", "agent1", "world")]


@pytest.mark.asyncio
async def test_fire_error():
    log = []

    async def on_error(agent_name, input_text, error):
        log.append(("error", agent_name, str(error)))

    hooks = LifecycleHooks(on_error=[on_error])
    await hooks.fire_error("agent1", "hello", ValueError("boom"))
    assert log == [("error", "agent1", "boom")]


@pytest.mark.asyncio
async def test_multiple_hooks():
    log = []

    async def hook_a(agent_name, input_text):
        log.append("a")

    async def hook_b(agent_name, input_text):
        log.append("b")

    hooks = LifecycleHooks(on_start=[hook_a, hook_b])
    await hooks.fire_start("x", "y")
    assert log == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_hooks():
    hooks = LifecycleHooks()
    await hooks.fire_start("x", "y")
    await hooks.fire_complete("x", "y", "z")
    await hooks.fire_error("x", "y", ValueError("e"))
