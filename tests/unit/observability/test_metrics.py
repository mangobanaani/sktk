# tests/unit/observability/test_metrics.py
import pytest

from sktk.core.types import TokenUsage
from sktk.observability.metrics import PricingModel, TokenTracker


@pytest.mark.asyncio
async def test_token_tracker_record_and_get():
    tracker = TokenTracker()
    await tracker.record(
        agent_name="analyst",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
    )
    usage = await tracker.get_usage(session_id="s1")
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50


@pytest.mark.asyncio
async def test_token_tracker_aggregates():
    tracker = TokenTracker()
    await tracker.record(
        agent_name="a",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
    )
    await tracker.record(
        agent_name="b",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
    )
    usage = await tracker.get_usage(session_id="s1")
    assert usage.prompt_tokens == 300
    assert usage.completion_tokens == 150


@pytest.mark.asyncio
async def test_token_tracker_filter_by_agent():
    tracker = TokenTracker()
    await tracker.record(
        agent_name="a",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
    )
    await tracker.record(
        agent_name="b",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
    )
    usage = await tracker.get_usage(session_id="s1", agent_name="a")
    assert usage.prompt_tokens == 100


@pytest.mark.asyncio
async def test_token_tracker_with_pricing():
    pricing = PricingModel(prices={"gpt-4": {"prompt": 0.03, "completion": 0.06}})
    tracker = TokenTracker(pricing=pricing)
    await tracker.record(
        agent_name="a",
        session_id="s1",
        model="gpt-4",
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
    )
    usage = await tracker.get_usage(session_id="s1")
    assert usage.total_cost_usd == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_token_tracker_unknown_session():
    tracker = TokenTracker()
    usage = await tracker.get_usage(session_id="nonexistent")
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0


def test_pricing_model_unknown_model():
    pricing = PricingModel(prices={"gpt-4": {"prompt": 0.03, "completion": 0.06}})
    cost = pricing.calculate("unknown-model", prompt_tokens=100, completion_tokens=50)
    assert cost is None


def test_record_metric_stores_samples():
    from sktk.observability.metrics import get_metric_samples, record_metric, reset_metrics

    reset_metrics()
    record_metric("agent.service_call.duration", 1.23, {"agent": "a1"})
    record_metric("agent.service_call.duration", 2.34, {"agent": "a2"})

    samples = get_metric_samples("agent.service_call.duration")
    assert len(samples) == 2
    assert samples[0]["value"] == 1.23
    assert samples[1]["tags"]["agent"] == "a2"


def test_get_metric_samples_returns_defensive_copies():
    from sktk.observability.metrics import get_metric_samples, record_metric, reset_metrics

    reset_metrics()
    record_metric("agent.service_call.duration", 1.23, {"agent": "a1"})

    samples = get_metric_samples("agent.service_call.duration")
    samples[0]["value"] = 9.87
    samples[0]["tags"]["agent"] = "mutated"

    reread = get_metric_samples("agent.service_call.duration")
    assert reread[0]["value"] == 1.23
    assert reread[0]["tags"]["agent"] == "a1"


def test_metrics_retention_caps_samples():
    from sktk.observability.metrics import (
        get_metric_samples,
        record_metric,
        reset_metrics,
        set_metrics_max_samples,
    )

    reset_metrics()
    set_metrics_max_samples(2)
    record_metric("m", 1, None)
    record_metric("m", 2, None)
    record_metric("m", 3, None)

    samples = get_metric_samples("m")
    assert [s["value"] for s in samples] == [2.0, 3.0]


def test_reset_metrics_restores_default_max_samples():
    from sktk.observability.metrics import (
        get_metric_samples,
        record_metric,
        reset_metrics,
        set_metrics_max_samples,
    )

    reset_metrics()
    set_metrics_max_samples(1)
    record_metric("m", 1, None)
    record_metric("m", 2, None)
    assert [s["value"] for s in get_metric_samples("m")] == [2.0]

    reset_metrics()
    record_metric("m", 1, None)
    record_metric("m", 2, None)
    assert [s["value"] for s in get_metric_samples("m")] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_token_tracker_filter_by_model():
    tracker = TokenTracker()
    await tracker.record("a", "s1", "gpt-4", TokenUsage(prompt_tokens=100, completion_tokens=50))
    await tracker.record("a", "s1", "gpt-3.5", TokenUsage(prompt_tokens=200, completion_tokens=100))
    usage = await tracker.get_usage(session_id="s1", model="gpt-4")
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50


@pytest.mark.asyncio
async def test_token_tracker_clear_all():
    tracker = TokenTracker()
    await tracker.record("a", "s1", "gpt-4", TokenUsage(prompt_tokens=100))
    await tracker.record("b", "s2", "gpt-4", TokenUsage(prompt_tokens=200))
    await tracker.clear()
    usage_s1 = await tracker.get_usage(session_id="s1")
    usage_s2 = await tracker.get_usage(session_id="s2")
    assert usage_s1.prompt_tokens == 0
    assert usage_s2.prompt_tokens == 0


@pytest.mark.asyncio
async def test_token_tracker_clear_by_session():
    tracker = TokenTracker()
    await tracker.record("a", "s1", "gpt-4", TokenUsage(prompt_tokens=100))
    await tracker.record("b", "s2", "gpt-4", TokenUsage(prompt_tokens=200))
    await tracker.clear(session_id="s1")
    usage_s1 = await tracker.get_usage(session_id="s1")
    usage_s2 = await tracker.get_usage(session_id="s2")
    assert usage_s1.prompt_tokens == 0
    assert usage_s2.prompt_tokens == 200


@pytest.mark.asyncio
async def test_token_tracker_trims_oldest_when_exceeding_max_records():
    tracker = TokenTracker(max_records=3)
    for i in range(5):
        await tracker.record("a", "s1", "gpt-4", TokenUsage(prompt_tokens=(i + 1) * 10))
    # Only the last 3 records should survive (prompt_tokens: 30, 40, 50)
    usage = await tracker.get_usage(session_id="s1")
    assert usage.prompt_tokens == 30 + 40 + 50
