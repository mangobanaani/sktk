"""Thread pool executor for CPU-bound work.

Wraps asyncio.to_thread and provides a managed thread pool
for running blocking operations without stalling the event loop.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_default_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_pool_max_workers: int = 0


def get_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create the default thread pool."""
    global _default_pool, _pool_max_workers
    with _pool_lock:
        if _default_pool is None:
            _default_pool = ThreadPoolExecutor(max_workers=max_workers)
            _pool_max_workers = max_workers
        elif _pool_max_workers != max_workers:
            logger.warning(
                "Thread pool already exists with max_workers=%d; ignoring requested max_workers=%d",
                _pool_max_workers,
                max_workers,
            )
        return _default_pool


async def run_in_executor(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a sync function in a thread pool without blocking the event loop.

    Usage:
        result = await run_in_executor(heavy_computation, data)
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        import functools

        bound = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(get_thread_pool(), bound)
    return await loop.run_in_executor(get_thread_pool(), fn, *args)


async def run_parallel(*tasks: Callable[..., T]) -> list[T]:
    """Run multiple sync functions in parallel using the thread pool.

    Usage:
        results = await run_parallel(fn1, fn2, fn3)
    """
    loop = asyncio.get_running_loop()
    pool = get_thread_pool()
    futures = [loop.run_in_executor(pool, task) for task in tasks]
    return list(await asyncio.gather(*futures))


def _cleanup_pool() -> None:
    """Atexit handler to release the global thread pool on interpreter shutdown."""
    global _default_pool
    with _pool_lock:
        if _default_pool is not None:
            _default_pool.shutdown(wait=False)
            _default_pool = None


atexit.register(_cleanup_pool)


def shutdown_pool(wait: bool = True) -> None:
    """Shut down the default thread pool."""
    global _default_pool, _pool_max_workers
    with _pool_lock:
        if _default_pool is not None:
            _default_pool.shutdown(wait=wait)
            _default_pool = None
            _pool_max_workers = 0
