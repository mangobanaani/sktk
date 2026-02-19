# tests/unit/agent/test_builder.py
import pytest

from sktk.agent.builder import build_safe_agent, default_safety_filters
from sktk.agent.providers import CompletionResult
from sktk.agent.router import LatencyPolicy, Router
from sktk.core.errors import GuardrailException
from sktk.core.types import TokenUsage


class DummyProvider:
    def __init__(self, reply: str):
        self._reply = reply
        self.latency_ms = 10
        self.cost = 0.001

    @property
    def name(self):
        return "dummy"

    async def complete(self, messages, **kwargs):
        return self._reply


class RichDummyProvider(DummyProvider):
    async def complete(self, messages, **kwargs):
        return CompletionResult(
            text=self._reply,
            usage=TokenUsage(prompt_tokens=2, completion_tokens=5),
            metadata={"model": "rich-dummy"},
        )


@pytest.mark.asyncio
async def test_default_safety_filters():
    filters = default_safety_filters()
    assert len(filters) >= 3


@pytest.mark.asyncio
async def test_build_safe_agent_uses_router():
    provider = DummyProvider("ok")
    router = Router([provider], policy=LatencyPolicy())
    agent = build_safe_agent(name="safe", instructions="Hi", router=router)
    result = await agent.invoke("hello")
    assert result == "ok"


@pytest.mark.asyncio
async def test_build_safe_agent_blocked_pattern():
    provider = DummyProvider("should not run")
    router = Router([provider], policy=LatencyPolicy())
    agent = build_safe_agent(
        name="safe", instructions="Hi", router=router, blocked_patterns=[r"forbidden"]
    )
    with pytest.raises(GuardrailException):
        await agent.invoke("forbidden content")


@pytest.mark.asyncio
async def test_build_safe_agent_stream_uses_router():
    provider = DummyProvider("streamed response")
    router = Router([provider], policy=LatencyPolicy())
    agent = build_safe_agent(name="safe", instructions="Hi", router=router)

    chunks: list[str] = []
    async for chunk in agent.invoke_stream("hello"):
        chunks.append(chunk)

    assert "".join(chunks).strip() == "streamed response"


@pytest.mark.asyncio
async def test_build_safe_agent_captures_provider_usage_metadata():
    provider = RichDummyProvider("ok")
    router = Router([provider], policy=LatencyPolicy())
    agent = build_safe_agent(name="safe", instructions="Hi", router=router)

    result = await agent.invoke("hello")

    assert result == "ok"
    assert agent._last_provider == "dummy"
    assert agent._last_usage is not None
    assert agent._last_usage.total_tokens == 7
    assert agent._last_response_metadata["model"] == "rich-dummy"
