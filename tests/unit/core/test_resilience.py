# tests/unit/core/test_resilience.py
import pytest

from sktk.core.errors import RetryExhaustedError
from sktk.core.resilience import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
)


@pytest.mark.asyncio
async def test_retry_succeeds_first_try():
    call_count = 0

    async def succeed():
        nonlocal call_count
        call_count += 1
        return "ok"

    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    result = await policy.execute(succeed)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    call_count = 0

    async def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "recovered"

    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    result = await policy.execute(fail_twice)
    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_exhausted():
    async def always_fail():
        raise ValueError("always")

    policy = RetryPolicy(max_retries=2, base_delay=0.01)
    with pytest.raises(RetryExhaustedError) as exc_info:
        await policy.execute(always_fail)
    assert exc_info.value.attempts == 3  # 1 initial + 2 retries = 3 attempts


@pytest.mark.asyncio
async def test_retry_fixed_backoff():
    policy = RetryPolicy(max_retries=1, base_delay=0.01, backoff=BackoffStrategy.FIXED)
    delay = policy._compute_delay(0)
    assert delay == 0.01
    delay = policy._compute_delay(5)
    assert delay == 0.01


@pytest.mark.asyncio
async def test_retry_exponential_backoff():
    policy = RetryPolicy(
        max_retries=3, base_delay=1.0, max_delay=10.0, backoff=BackoffStrategy.EXPONENTIAL
    )
    assert policy._compute_delay(0) == 1.0
    assert policy._compute_delay(1) == 2.0
    assert policy._compute_delay(2) == 4.0
    assert policy._compute_delay(10) == 10.0  # capped at max_delay


@pytest.mark.asyncio
async def test_retry_exponential_jitter():
    policy = RetryPolicy(max_retries=3, base_delay=1.0, backoff=BackoffStrategy.EXPONENTIAL_JITTER)
    delays = [policy._compute_delay(2) for _ in range(10)]
    # Should have some variance due to jitter
    assert len(set(delays)) > 1


@pytest.mark.asyncio
async def test_retry_only_retryable_exceptions():
    async def fail_type_error():
        raise TypeError("wrong type")

    policy = RetryPolicy(max_retries=3, base_delay=0.01, retryable_exceptions=(ValueError,))
    with pytest.raises(TypeError):
        await policy.execute(fail_type_error)


def test_retry_policy_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries"):
        RetryPolicy(max_retries=-1)


def test_circuit_breaker_starts_closed():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures():
    cb = CircuitBreaker(failure_threshold=3)

    async def fail():
        raise ValueError("boom")

    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.execute(fail)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_transitions_to_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)

    async def fail():
        raise ValueError("boom")

    async def succeed():
        return "ok"

    with pytest.raises(ValueError):
        await cb.execute(fail)
    assert cb._state == CircuitState.OPEN

    # The OPEN -> HALF_OPEN transition happens inside execute() under the lock.
    # With recovery_timeout=0.0, the next execute() call will transition to
    # HALF_OPEN, allow the test call, and then transition to CLOSED on success.
    result = await cb.execute(succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_success_resets():
    cb = CircuitBreaker(failure_threshold=3)

    async def fail():
        raise ValueError("boom")

    async def succeed():
        return "ok"

    with pytest.raises(ValueError):
        await cb.execute(fail)
    with pytest.raises(ValueError):
        await cb.execute(fail)
    await cb.execute(succeed)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open():
    cb = CircuitBreaker(failure_threshold=1)

    async def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await cb.execute(fail)
    assert cb.state == CircuitState.OPEN

    async def fn():
        return "ok"

    with pytest.raises(RuntimeError, match="OPEN"):
        await cb.execute(fn)


@pytest.mark.asyncio
async def test_circuit_breaker_execute_success():
    cb = CircuitBreaker()

    async def fn():
        return "ok"

    result = await cb.execute(fn)
    assert result == "ok"


@pytest.mark.asyncio
async def test_circuit_breaker_execute_records_failure():
    cb = CircuitBreaker(failure_threshold=2)

    async def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await cb.execute(fail)
    assert cb._failure_count == 1


def test_retry_policy_rejects_negative_base_delay():
    with pytest.raises(ValueError, match="base_delay"):
        RetryPolicy(base_delay=-1)


def test_retry_policy_rejects_negative_max_delay():
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(max_delay=-1)


def test_retry_policy_accepts_zero_base_delay():
    policy = RetryPolicy(base_delay=0)
    assert policy.base_delay == 0


def test_retry_policy_accepts_zero_max_delay():
    policy = RetryPolicy(max_delay=0)
    assert policy.max_delay == 0
