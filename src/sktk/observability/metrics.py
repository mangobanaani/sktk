"""Token usage tracking and pricing models."""

from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import dataclass
from typing import Any

from sktk.core.types import TokenUsage

_metrics_lock = threading.Lock()
_metrics: dict[str, list[dict[str, Any]]] = {}
DEFAULT_METRICS_MAX_SAMPLES = 10_000
_metrics_max_samples = DEFAULT_METRICS_MAX_SAMPLES


def set_metrics_max_samples(max_samples: int) -> None:
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    global _metrics_max_samples
    with _metrics_lock:
        _metrics_max_samples = max_samples


def record_metric(name: str, value: float, tags: dict[str, Any] | None = None) -> None:
    """Record a metric sample.

    This is a lightweight in-memory backend that keeps runtime code decoupled
    from any specific metrics service.
    """
    with _metrics_lock:
        samples = _metrics.setdefault(name, [])
        samples.append({"value": float(value), "tags": dict(tags or {})})
        if len(samples) > _metrics_max_samples:
            del samples[: len(samples) - _metrics_max_samples]


def get_metric_samples(name: str) -> list[dict[str, Any]]:
    with _metrics_lock:
        return copy.deepcopy(_metrics.get(name, []))


def reset_metrics() -> None:
    with _metrics_lock:
        _metrics.clear()
        global _metrics_max_samples
        _metrics_max_samples = DEFAULT_METRICS_MAX_SAMPLES


@dataclass
class PricingModel:
    """Simple per-1k token pricing."""

    prices: dict[str, dict[str, float]]

    def calculate(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> float | None:
        pricing = self.prices.get(model)
        if not pricing:
            return None
        prompt_rate = pricing.get("prompt", 0.0)
        completion_rate = pricing.get("completion", 0.0)
        return (prompt_tokens / 1000.0) * prompt_rate + (
            completion_tokens / 1000.0
        ) * completion_rate


class TokenTracker:
    """Aggregate token usage across sessions and agents."""

    def __init__(self, pricing: PricingModel | None = None, max_records: int = 10_000) -> None:
        self._pricing = pricing
        self._max_records = max_records
        self._lock = asyncio.Lock()
        self._records: list[dict[str, Any]] = []

    async def record(
        self,
        agent_name: str,
        session_id: str,
        model: str,
        usage: TokenUsage,
    ) -> None:
        async with self._lock:
            self._records.append(
                {
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "model": model,
                    "usage": usage,
                }
            )
            if len(self._records) > self._max_records:
                del self._records[: len(self._records) - self._max_records]

    async def get_usage(
        self,
        session_id: str,
        agent_name: str | None = None,
        model: str | None = None,
    ) -> TokenUsage:
        async with self._lock:
            records = [
                r
                for r in self._records
                if r["session_id"] == session_id
                and (agent_name is None or r["agent_name"] == agent_name)
                and (model is None or r["model"] == model)
            ]
        total = TokenUsage()
        total_cost = 0.0
        cost_seen = False
        for rec in records:
            usage = rec["usage"]
            total = total + usage
            if usage.total_cost_usd is not None:
                total_cost += usage.total_cost_usd
                cost_seen = True
            elif self._pricing:
                cost = self._pricing.calculate(
                    rec["model"], usage.prompt_tokens, usage.completion_tokens
                )
                if cost is not None:
                    total_cost += cost
                    cost_seen = True
        if cost_seen:
            total.total_cost_usd = total_cost
        return total

    async def clear(self, session_id: str | None = None) -> None:
        async with self._lock:
            if session_id is None:
                self._records = []
            else:
                self._records = [r for r in self._records if r["session_id"] != session_id]
