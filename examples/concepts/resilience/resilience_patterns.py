"""Resilience: retry policies and circuit breakers.

Production LLM calls fail -- rate limits, timeouts, transient errors.
SKTK provides RetryPolicy (with configurable backoff) and CircuitBreaker
so your agents recover gracefully.

Usage:
    python examples/concepts/resilience/resilience_patterns.py
"""

import asyncio

from sktk import BackoffStrategy, RetryPolicy
from sktk.core.errors import RetryExhaustedError
from sktk.core.resilience import CircuitBreaker, CircuitState

# -- Helpers that simulate flaky LLM calls --

call_count = 0


async def flaky_llm_call(prompt: str) -> str:
    """Succeeds on the 3rd attempt."""
    global call_count  # noqa: PLW0603
    call_count += 1
    if call_count < 3:
        raise TimeoutError(f"attempt {call_count}: request timed out")
    return f"Response to: {prompt}"


async def always_failing(prompt: str) -> str:
    raise ConnectionError("service unavailable")


async def main() -> None:
    # -- 1) RetryPolicy with exponential backoff + jitter --
    print("=== Retry Policy (exponential jitter) ===")
    policy = RetryPolicy(
        max_retries=5,
        base_delay=0.01,  # fast for demo; use 1.0+ in production
        backoff=BackoffStrategy.EXPONENTIAL_JITTER,
        retryable_exceptions=(TimeoutError,),
    )

    global call_count  # noqa: PLW0603
    call_count = 0
    result = await policy.execute(flaky_llm_call, "Hello")
    print(f"  Succeeded after {call_count} attempts: {result!r}")

    # -- 2) RetryPolicy exhaustion --
    print("\n=== Retry Exhaustion ===")
    strict_policy = RetryPolicy(
        max_retries=2,
        base_delay=0.01,
        retryable_exceptions=(ConnectionError,),
    )
    try:
        await strict_policy.execute(always_failing, "test")
    except RetryExhaustedError as e:
        print(f"  Gave up after {e.attempts} retries: {e.last_error!r}")

    # -- 3) Circuit breaker --
    print("\n=== Circuit Breaker ===")
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    print(f"  Initial state: {cb.state.value}")

    # Trip the breaker with repeated failures
    for i in range(4):
        try:
            await cb.execute(always_failing, "test")
        except (ConnectionError, RuntimeError) as e:
            print(f"  Call {i + 1}: {type(e).__name__} (state={cb.state.value})")

    # Wait for recovery timeout, state transitions to HALF_OPEN
    await asyncio.sleep(0.15)
    print(f"\n  After recovery timeout: {cb.state.value}")

    # A successful call closes the circuit
    call_count = 2  # next flaky_llm_call will succeed
    result = await cb.execute(flaky_llm_call, "recovery")
    print(f"  Recovery call succeeded: {result!r}")
    print(f"  Final state: {cb.state.value}")
    assert cb.state == CircuitState.CLOSED


if __name__ == "__main__":
    asyncio.run(main())
