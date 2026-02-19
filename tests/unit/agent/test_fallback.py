# tests/unit/agent/test_fallback.py
import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.fallback import FallbackChain
from sktk.testing.mocks import MockKernel


@pytest.mark.asyncio
async def test_fallback_uses_first_agent():
    a = SKTKAgent(name="primary", instructions="test")
    kernel = MockKernel()
    kernel.expect_chat_completion(["primary result"])
    a.kernel = kernel

    b = SKTKAgent(name="backup", instructions="test")
    chain = FallbackChain([a, b])
    result = await chain.invoke("hello")
    assert result == "primary result"


@pytest.mark.asyncio
async def test_fallback_to_second():
    a = SKTKAgent(name="primary", instructions="test")
    # No kernel set, will raise NotImplementedError

    b = SKTKAgent(name="backup", instructions="test")
    kernel = MockKernel()
    kernel.expect_chat_completion(["backup result"])
    b.kernel = kernel

    chain = FallbackChain([a, b])
    result = await chain.invoke("hello")
    assert result == "backup result"


@pytest.mark.asyncio
async def test_fallback_all_fail():
    a = SKTKAgent(name="primary", instructions="test")
    b = SKTKAgent(name="backup", instructions="test")
    # Neither has a kernel

    chain = FallbackChain([a, b])
    with pytest.raises(NotImplementedError):
        await chain.invoke("hello")


def test_fallback_requires_agents():
    with pytest.raises(ValueError, match="at least one"):
        FallbackChain([])


def test_fallback_agents_property():
    a = SKTKAgent(name="a", instructions="test")
    b = SKTKAgent(name="b", instructions="test")
    chain = FallbackChain([a, b])
    assert len(chain.agents) == 2


@pytest.mark.asyncio
async def test_fallback_specific_exceptions():
    a = SKTKAgent(name="primary", instructions="test")
    # Will raise NotImplementedError

    b = SKTKAgent(name="backup", instructions="test")
    kernel = MockKernel()
    kernel.expect_chat_completion(["ok"])
    b.kernel = kernel

    chain = FallbackChain([a, b], fallback_exceptions=(NotImplementedError,))
    result = await chain.invoke("hello")
    assert result == "ok"


@pytest.mark.asyncio
async def test_fallback_warning_includes_exc_info(monkeypatch):
    calls = []

    class FakeLogger:
        def warning(self, msg, **kwargs):
            calls.append((msg, kwargs))

    a = SKTKAgent(name="primary", instructions="test")
    b = SKTKAgent(name="backup", instructions="test")
    kernel = MockKernel()
    kernel.expect_chat_completion(["ok"])
    b.kernel = kernel

    monkeypatch.setattr("sktk.agent.fallback._logger", FakeLogger())

    chain = FallbackChain([a, b], fallback_exceptions=(NotImplementedError,))
    result = await chain.invoke("hello")

    assert result == "ok"
    assert calls
    assert calls[0][1]["exc_info"].__class__.__name__ == "NotImplementedError"
