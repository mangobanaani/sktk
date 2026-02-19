"""Provider router with pluggable selection policies."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from secrets import SystemRandom
from typing import Any, Callable, Sequence

from sktk.agent.providers import LLMProvider, normalize_completion_result
from sktk.core.errors import CircuitBreakerOpenError
from sktk.core.resilience import CircuitBreaker, RetryPolicy

logger = logging.getLogger(__name__)


class RouterPolicy:
    """Strategy interface for selecting the next provider."""

    async def choose(self, providers: Sequence[LLMProvider]) -> LLMProvider:
        """Choose one provider from the available set."""
        raise NotImplementedError


class LatencyPolicy(RouterPolicy):
    """Pick provider with lowest known latency_ms attribute; fallback to first."""

    async def choose(self, providers: Sequence[LLMProvider]) -> LLMProvider:
        return min(providers, key=lambda p: getattr(p, "latency_ms", float("inf")))


class CostPolicy(RouterPolicy):
    """Pick provider with lowest known cost attribute; fallback to first."""

    async def choose(self, providers: Sequence[LLMProvider]) -> LLMProvider:
        return min(providers, key=lambda p: getattr(p, "cost", float("inf")))


class ABPolicy(RouterPolicy):
    """Pick variant with probability split; selector returns float in [0,1)."""

    def __init__(self, split: float = 0.5, selector: Callable[[], float] | None = None) -> None:
        self._split = split
        sysrand = SystemRandom()
        self._selector = selector or (lambda: sysrand.random())

    async def choose(self, providers: Sequence[LLMProvider]) -> LLMProvider:
        if len(providers) < 2:
            return providers[0]
        return providers[0] if self._selector() < self._split else providers[1]


class FallbackPolicy(RouterPolicy):
    """Always try in order; errors handled by Router fallback."""

    async def choose(self, providers: Sequence[LLMProvider]) -> LLMProvider:
        return providers[0]


@dataclass
class Router:
    """Routes completion requests to providers using a selection policy.

    Optionally wraps each provider call with a :class:`RetryPolicy` and/or
    :class:`CircuitBreaker` from ``sktk.core.resilience``.  When both are
    configured the retry policy wraps the circuit breaker which wraps the
    raw provider call.  If the circuit breaker is open for a given provider
    the router skips to the next provider immediately.

    Circuit breakers can be configured per-provider via ``circuit_breakers``
    (a dict keyed by provider name) or as a single shared fallback via
    ``circuit_breaker``.  Per-provider breakers take precedence; the shared
    breaker is used for any provider without a dedicated entry.
    """

    providers: Sequence[LLMProvider]
    policy: RouterPolicy
    retry_policy: RetryPolicy | None = field(default=None)
    circuit_breaker: CircuitBreaker | None = field(default=None)
    circuit_breakers: dict[str, CircuitBreaker] | None = field(default=None)

    def __post_init__(self) -> None:
        names = [getattr(p, "name", None) for p in self.providers]
        duplicates = {name for name in names if name is not None and names.count(name) > 1}
        if duplicates:
            dup_list = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate provider name(s): {dup_list}")

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        result, _ = await self.complete_with_metadata(messages, **kwargs)
        return result

    async def complete_with_metadata(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        timeout = kwargs.get("timeout")
        # Remove timeout from kwargs passed to provider
        provider_kwargs = {k: v for k, v in kwargs.items() if k != "timeout"}
        errors: list[Exception] = []
        tried: set[int] = set()
        # Try until one succeeds or all tried
        while len(tried) < len(self.providers):
            remaining = [(i, p) for i, p in enumerate(self.providers) if i not in tried]
            provider = await self.policy.choose([p for _, p in remaining])
            # Try identity match first (handles most cases)
            idx = next((i for i, p in remaining if p is provider), None)
            # Fall back to name match for wrapped/proxied providers
            if idx is None:
                idx = next(
                    (
                        i
                        for i, p in remaining
                        if getattr(p, "name", None) == getattr(provider, "name", None)
                    ),
                    None,
                )
            if idx is None:
                logger.warning("Router: policy returned unknown provider, falling back")
                idx = remaining[0][0]
            provider = next(p for i, p in remaining if i == idx)
            tried.add(idx)
            logger.debug("Router selected provider %s", provider.name)
            try:
                raw_result = await self._call_provider(
                    provider, messages, timeout, **provider_kwargs
                )
                result = normalize_completion_result(raw_result)
                meta: dict[str, Any] = {"provider": provider.name, **result.metadata}
                if result.usage is not None:
                    meta["usage"] = result.usage
                return result.text, meta
            except CircuitBreakerOpenError as e:
                logger.debug("Skipping provider %s: circuit breaker open", provider.name)
                errors.append(e)
                continue
            except Exception as e:
                logger.debug("Provider %s failed, falling back to next: %s", provider.name, e)
                errors.append(e)
                continue
        # If all failed, raise last
        if errors:
            raise errors[-1]
        raise RuntimeError("Router has no providers")

    async def _call_provider(
        self,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        timeout: float | None,
        **kwargs: Any,
    ) -> Any:
        """Call a single provider, optionally wrapped by circuit breaker and retry."""

        async def _raw_call() -> Any:
            call = provider.complete(messages, **kwargs)
            if timeout is None:
                return await call
            return await asyncio.wait_for(call, timeout=float(timeout))

        # Build the callable chain: retry wraps circuit_breaker wraps raw call
        target = _raw_call

        # Per-provider breaker takes precedence, then shared fallback
        cb: CircuitBreaker | None = None
        if self.circuit_breakers is not None and provider.name in self.circuit_breakers:
            cb = self.circuit_breakers[provider.name]
        elif self.circuit_breaker is not None:
            cb = self.circuit_breaker

        if cb is not None:
            _cb = cb  # capture for closure
            _inner = target  # snapshot before reassignment

            async def _cb_call() -> Any:
                return await _cb.execute(_inner)

            target = _cb_call

        if self.retry_policy is not None:
            return await self.retry_policy.execute(target)

        return await target()
