"""Token quota and cost control.

Limits token consumption per user, session, or time period
with automatic tracking and enforcement.
"""

from __future__ import annotations

import asyncio
import bisect
import time
from dataclasses import dataclass, field

from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Deny, FilterResult


@dataclass
class TokenQuota:
    """Track and enforce token usage limits.

    Uses an asyncio lock to prevent concurrent access from causing
    undercounting when shared across coroutines.

    Usage:
        quota = TokenQuota(max_tokens=100000, window_seconds=3600)
        await quota.record_usage("user1", 500)
        remaining = await quota.remaining("user1")
    """

    max_tokens: int = 100_000
    window_seconds: float = 3600.0
    _usage: dict[str, list[tuple[float, int]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        if key in self._usage:
            entries = self._usage[key]
            idx = bisect.bisect_right(entries, cutoff, key=lambda e: e[0])
            if idx:
                del entries[:idx]
            if not entries:
                del self._usage[key]

    async def record_usage(self, key: str, tokens: int) -> None:
        """Record token usage for a key (user_id, session_id, etc)."""
        async with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            if key not in self._usage:
                self._usage[key] = []
            self._usage[key].append((now, tokens))

    async def used(self, key: str) -> int:
        """Get total tokens used in current window."""
        async with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            return sum(n for _, n in self._usage.get(key, []))

    async def remaining(self, key: str) -> int:
        """Get remaining tokens in quota.

        Note: This is advisory only. The result may be stale by the time
        the caller acts on it. Use try_consume() for atomic check-and-subtract.
        """
        async with self._lock:
            self._prune(key, time.monotonic())
            used = sum(u for _, u in self._usage.get(key, []))
        return max(0, self.max_tokens - used)

    async def is_exceeded(self, key: str) -> bool:
        """Check if quota is exceeded."""
        return await self.used(key) >= self.max_tokens

    async def try_consume(self, key: str, tokens: int) -> bool:
        """Atomically check budget and record usage. Returns True if consumed, False if over budget."""
        async with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            used = sum(u for _, u in self._usage.get(key, []))
            if used + tokens > self.max_tokens:
                return False
            self._usage.setdefault(key, []).append((now, tokens))
            return True


class TokenQuotaFilter:
    """Filter that enforces token quotas by tracking combined input + output tokens.

    The quota budget (``max_tokens`` on the underlying :class:`TokenQuota`)
    should be set to the **total** token spend you want to allow, because
    both input and output tokens count against it:

    * ``on_input`` atomically checks the budget and records estimated input
      tokens via ``try_consume``.
    * ``on_output`` records estimated output tokens via ``record_usage``.

    This means a request that uses 500 input tokens and 300 output tokens
    will consume 800 tokens from the quota. Set ``max_tokens`` accordingly
    (i.e. to the combined input + output budget, not just one side).

    Usage:
        quota = TokenQuota(max_tokens=10000, window_seconds=3600)
        filter = TokenQuotaFilter(quota=quota, key_field="user_id")
        agent = SKTKAgent(name="a", filters=[filter])
    """

    def __init__(
        self,
        quota: TokenQuota,
        key_field: str = "session_id",
        tokens_per_word: float = 1.3,
    ) -> None:
        self._quota = quota
        self._key_field = key_field
        self._tokens_per_word = tokens_per_word

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from word count using a fixed multiplier."""
        return int(len(text.split()) * self._tokens_per_word)

    def _get_key(self, context: FilterContext) -> str:
        """Extract the quota key (e.g. user_id, session_id) from filter context."""
        return context.metadata.get(self._key_field, "default")

    async def on_input(self, context: FilterContext) -> FilterResult:
        """Deny input if projected token usage would exceed the combined quota.

        Estimated input tokens are atomically checked and recorded against
        the quota via ``try_consume``.  Together with the output tokens
        recorded by :meth:`on_output`, this enforces a combined
        input + output budget.
        """
        key = self._get_key(context)
        tokens = self._estimate_tokens(context.content)
        if not await self._quota.try_consume(key, tokens):
            used = await self._quota.used(key)
            return Deny(reason=f"Token quota exceeded: {used}/{self._quota.max_tokens}")
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        """Record estimated output token usage against the combined quota.

        Output tokens are added to the same quota budget that input tokens
        were charged against in :meth:`on_input`, so the quota reflects
        total (input + output) token spend.
        """
        key = self._get_key(context)
        tokens = self._estimate_tokens(context.content)
        await self._quota.record_usage(key, tokens)
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return Allow()
