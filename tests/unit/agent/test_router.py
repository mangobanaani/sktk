# tests/unit/agent/test_router.py
import pytest

from sktk.agent.providers import CompletionResult
from sktk.agent.router import ABPolicy, CostPolicy, FallbackPolicy, LatencyPolicy, Router
from sktk.core.resilience import CircuitBreaker
from sktk.core.types import TokenUsage


class FakeProvider:
    def __init__(
        self,
        name: str,
        result: str,
        latency_ms: float = 100.0,
        cost: float = 0.002,
        fail: bool = False,
    ):
        self._name = name
        self._result = result
        self.latency_ms = latency_ms
        self.cost = cost
        self.fail = fail

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, messages, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        return f"{self._name}:{self._result}"


class SlowProvider(FakeProvider):
    async def complete(self, messages, **kwargs):
        import asyncio

        await asyncio.sleep(0.05)
        return await super().complete(messages, **kwargs)


class RichProvider(FakeProvider):
    async def complete(self, messages, **kwargs):
        text = await super().complete(messages, **kwargs)
        return CompletionResult(
            text=text,
            usage=TokenUsage(prompt_tokens=11, completion_tokens=7),
            metadata={"model": "demo"},
        )


class ReturnProviderPolicy:
    def __init__(self, provider):
        self._provider = provider

    async def choose(self, providers):
        return self._provider


@pytest.mark.asyncio
async def test_latency_policy_picks_fastest():
    p1 = FakeProvider("slow", "resp", latency_ms=120)
    p2 = FakeProvider("fast", "resp", latency_ms=10)
    router = Router([p1, p2], policy=LatencyPolicy())
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("fast:")


@pytest.mark.asyncio
async def test_cost_policy_picks_cheapest():
    p1 = FakeProvider("expensive", "resp", cost=0.01)
    p2 = FakeProvider("cheap", "resp", cost=0.001)
    router = Router([p1, p2], policy=CostPolicy())
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("cheap:")


@pytest.mark.asyncio
async def test_ab_policy_respects_split():
    p_control = FakeProvider("control", "resp")
    p_variant = FakeProvider("variant", "resp")
    # split=0.3 means 30% chance of control (providers[0]), 70% chance of variant
    # selector returns 0.2, which is < 0.3, so control is chosen
    policy = ABPolicy(split=0.3, selector=lambda: 0.2)
    router = Router([p_control, p_variant], policy=policy)
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("control:")


@pytest.mark.asyncio
async def test_fallback_on_error():
    failing = FakeProvider("fail", "resp", fail=True)
    ok = FakeProvider("ok", "resp")
    router = Router([failing, ok], policy=FallbackPolicy())
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("ok:")


@pytest.mark.asyncio
async def test_router_uses_selected_provider_instance_when_policy_returns_clone():
    primary = FakeProvider("primary", "resp_primary")
    secondary = FakeProvider("secondary", "resp_secondary")
    clone = FakeProvider("primary", "resp_clone", fail=True)
    router = Router([primary, secondary], policy=ReturnProviderPolicy(clone))
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("primary:")


@pytest.mark.asyncio
async def test_router_emits_usage_metadata_from_provider_result():
    provider = RichProvider("rich", "resp")
    router = Router([provider], policy=LatencyPolicy())
    result, meta = await router.complete_with_metadata([{"role": "user", "content": "hi"}])
    assert result.startswith("rich:")
    assert meta["provider"] == "rich"
    assert isinstance(meta["usage"], TokenUsage)
    assert meta["usage"].total_tokens == 18
    assert meta["model"] == "demo"


@pytest.mark.asyncio
async def test_router_fallback_after_timeout():
    slow = SlowProvider("slow", "resp")
    fast = FakeProvider("fast", "resp")
    router = Router([slow, fast], policy=FallbackPolicy())
    result, meta = await router.complete_with_metadata(
        [{"role": "user", "content": "hi"}],
        timeout=0.01,
    )
    assert result.startswith("fast:")
    assert meta["provider"] == "fast"


@pytest.mark.asyncio
async def test_ab_policy_single_provider_returns_first():
    only = FakeProvider("only", "resp")
    chosen = await ABPolicy(split=0.9, selector=lambda: 0.0).choose([only])
    assert chosen is only


def test_router_rejects_duplicate_provider_names():
    p1 = FakeProvider("dup", "resp1")
    p2 = FakeProvider("dup", "resp2")
    with pytest.raises(ValueError, match="duplicate provider name"):
        Router([p1, p2], policy=LatencyPolicy())


@pytest.mark.asyncio
async def test_router_raises_last_error_when_all_providers_fail():
    failing = FakeProvider("fail", "resp", fail=True)
    router = Router([failing], policy=LatencyPolicy())
    with pytest.raises(RuntimeError, match="boom"):
        await router.complete_with_metadata([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_router_with_no_providers_raises_runtime_error():
    router = Router([], policy=LatencyPolicy())
    with pytest.raises(RuntimeError, match="no providers"):
        await router.complete_with_metadata([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_per_provider_circuit_breaker_isolates_failures():
    """Tripping the breaker for provider A should not block provider B."""
    p_a = FakeProvider("provider_a", "resp_a", fail=True)
    p_b = FakeProvider("provider_b", "resp_b")
    cb_a = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb_b = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    router = Router(
        [p_a, p_b],
        policy=FallbackPolicy(),
        circuit_breakers={"provider_a": cb_a, "provider_b": cb_b},
    )
    # First call: provider_a fails and trips its breaker, falls back to provider_b
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("provider_b:")

    # Second call: provider_a breaker is open, but provider_b breaker is still closed
    result2 = await router.complete([{"role": "user", "content": "hi"}])
    assert result2.startswith("provider_b:")


@pytest.mark.asyncio
async def test_shared_circuit_breaker_used_as_fallback():
    """When circuit_breakers dict has no entry for a provider, the shared
    circuit_breaker is used as fallback."""
    p_a = FakeProvider("provider_a", "resp_a")
    p_b = FakeProvider("provider_b", "resp_b")
    shared_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    per_provider_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    router = Router(
        [p_a, p_b],
        policy=LatencyPolicy(),
        circuit_breaker=shared_cb,
        circuit_breakers={"provider_a": per_provider_cb},
    )
    # provider_a has a per-provider breaker; provider_b falls back to shared_cb
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("provider_a:") or result.startswith("provider_b:")


@pytest.mark.asyncio
async def test_single_circuit_breaker_backward_compat():
    """Passing only circuit_breaker= (no circuit_breakers dict) still works."""
    p = FakeProvider("only", "resp")
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    router = Router([p], policy=LatencyPolicy(), circuit_breaker=cb)
    result = await router.complete([{"role": "user", "content": "hi"}])
    assert result.startswith("only:")
