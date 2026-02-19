# tests/unit/core/test_executor.py
import pytest

from sktk.core.executor import get_thread_pool, run_in_executor, run_parallel, shutdown_pool


@pytest.mark.asyncio
async def test_run_in_executor():
    def add(a, b):
        return a + b

    result = await run_in_executor(add, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_run_in_executor_with_kwargs():
    def greet(name, greeting="hello"):
        return f"{greeting} {name}"

    result = await run_in_executor(greet, "world", greeting="hi")
    assert result == "hi world"


@pytest.mark.asyncio
async def test_run_parallel():
    def double(x):
        return x * 2

    def triple(x):
        return x * 3

    results = await run_parallel(lambda: double(5), lambda: triple(5))
    assert sorted(results) == [10, 15]


def test_get_thread_pool():
    shutdown_pool()  # Reset state
    pool = get_thread_pool(max_workers=2)
    assert pool is not None
    pool2 = get_thread_pool()
    assert pool is pool2  # Same instance
    shutdown_pool()


def test_shutdown_pool():
    get_thread_pool()
    shutdown_pool()
    # After shutdown, a new call creates a fresh pool
    pool = get_thread_pool()
    assert pool is not None
    shutdown_pool()
