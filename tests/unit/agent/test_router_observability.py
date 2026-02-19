# tests/unit/agent/test_router_observability.py
import pytest

from sktk.agent.providers import CompletionResult
from sktk.agent.router import LatencyPolicy, Router
from sktk.core.events import MessageEvent
from sktk.core.types import TokenUsage


class FakeProvider:
    def __init__(self, name: str, reply: str):
        self._name = name
        self.reply = reply
        self.latency_ms = 10
        self.cost = 0.001

    @property
    def name(self):
        return self._name

    async def complete(self, messages, **kwargs):
        return CompletionResult(
            text=self.reply,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            metadata={"model": "obs"},
        )


@pytest.mark.asyncio
async def test_router_returns_choice_metadata():
    p = FakeProvider("p1", "ok")
    router = Router([p], policy=LatencyPolicy())
    result, meta = await router.complete_with_metadata([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert meta["provider"] == "p1"
    assert meta["usage"].total_tokens == 3
    assert meta["model"] == "obs"


def test_message_event_can_carry_provider():
    ev = MessageEvent(
        agent="a",
        role="assistant",
        content="hi",
        token_usage=None,
        correlation_id="c",
        timestamp=None,
        provider="p1",
    )
    assert ev.provider == "p1"
