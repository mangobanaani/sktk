from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Awaitable, Callable

MetricsHook = Callable[[str, dict[str, Any]], Awaitable[None] | None]


async def _emit_metrics(hook: MetricsHook | None, event: str, payload: dict[str, Any]) -> None:
    """Fire metrics hook if provided; supports sync or async callables."""
    if hook is None:
        return
    try:
        result = hook(event, payload)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        # Metrics should never break core flows.
        return


class MetricsDispatcher:
    """Async queue-based metrics dispatcher."""

    def __init__(self, hook: MetricsHook | None, max_queue: int = 1000) -> None:
        self._hook = hook
        self._queue: deque[tuple[str, dict[str, Any]]] = deque()
        self._max_queue = max_queue
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._hook is None:
            return
        if len(self._queue) >= self._max_queue:
            self._queue.popleft()
        self._queue.append((event, payload))

    async def _run(self) -> None:
        while True:
            if not self._queue:
                await asyncio.sleep(0.01)
                continue
            event, payload = self._queue.popleft()
            await _emit_metrics(self._hook, event, payload)
