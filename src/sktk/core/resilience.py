"""Retry and resilience patterns for LLM calls.

Provides exponential backoff, jitter, and circuit breaker patterns
for production-grade reliability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from secrets import SystemRandom
from typing import Any, Awaitable, Callable, TypeVar

from sktk.core.errors import CircuitBreakerOpenError, RetryExhaustedError

logger = logging.getLogger(__name__)

T = TypeVar("T")
_SYSRAND = SystemRandom()


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass(frozen=True)
class RetryPolicy:
    """Configurable retry policy with backoff.

    Usage:
        policy = RetryPolicy(max_retries=3, backoff=BackoffStrategy.EXPONENTIAL_JITTER)
        result = await policy.execute(some_async_fn, "arg1", key="val")
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")

    def _compute_delay(self, attempt: int) -> float:
        if self.backoff == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.backoff == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2**attempt)
        else:  # EXPONENTIAL_JITTER
            delay = self.base_delay * (2**attempt)
            delay = delay * (0.5 + _SYSRAND.random() * 0.5)
        return min(delay, self.max_delay)

    async def execute(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute fn with retries according to this policy."""
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except CircuitBreakerOpenError:
                raise
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.debug(
                        "Retry %d/%d after %.2fs for %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
        assert last_exception is not None  # guaranteed by loop structure
        logger.warning(
            "Retry exhausted after %d attempts, last error: %s",
            self.max_retries + 1,
            last_exception,
        )
        raise RetryExhaustedError(
            attempts=self.max_retries + 1,
            last_error=last_exception,
        )


class CircuitState(Enum):
    """Circuit-breaker states describing failure handling mode."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker to avoid hammering a failing service.

    - CLOSED: normal operation, failures counted
    - OPEN: all calls rejected immediately
    - HALF_OPEN: one test call allowed to check recovery
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    exception_types: tuple[type[BaseException], ...] = (Exception,)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_in_flight: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state.

        Note: The actual OPEN -> HALF_OPEN transition happens inside
        ``execute()`` under the lock. This property is a snapshot read.
        """
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN if the recovery timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            logger.info("Circuit breaker transitioning OPEN -> HALF_OPEN")
            self._state = CircuitState.HALF_OPEN

    def _record_success(self) -> None:
        """Record a successful call. Must be called under self._lock."""
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker transitioning %s -> CLOSED", self._state.value.upper())
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record a failed call. Must be called under self._lock."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        logger.debug(
            "Circuit breaker recorded failure %d/%d", self._failure_count, self.failure_threshold
        )
        if self._failure_count >= self.failure_threshold:
            logger.info("Circuit breaker transitioning CLOSED -> OPEN")
            self._state = CircuitState.OPEN

    async def execute(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute fn through the circuit breaker."""
        async with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError()
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight:
                    raise CircuitBreakerOpenError()
                self._half_open_in_flight = True
        try:
            result = await fn(*args, **kwargs)
            async with self._lock:
                self._half_open_in_flight = False
                self._record_success()
            return result
        except Exception as exc:
            if isinstance(exc, self.exception_types):
                async with self._lock:
                    self._half_open_in_flight = False
                    self._record_failure()
                raise
            # Non-matching exceptions should not trip the breaker
            async with self._lock:
                self._half_open_in_flight = False
            raise
