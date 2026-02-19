"""Middleware pattern for agent invocations.

Middleware wraps agent calls to add cross-cutting concerns like
timing, caching, logging, and error handling without modifying agents.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

InvokeNext = Callable[[str], Awaitable[Any]]
Middleware = Callable[[str, str, InvokeNext], Awaitable[Any]]


class MiddlewareStack:
    """Composable middleware stack for agent invocations.

    Usage:
        stack = MiddlewareStack()

        @stack.use
        async def timing(agent_name, message, next_fn):
            start = time.monotonic()
            result = await next_fn(message)
            print(f"{agent_name} took {time.monotonic() - start:.3f}s")
            return result

        # Apply to agent
        agent = SKTKAgent(name="a", instructions="...")
        wrapped_invoke = stack.wrap(agent.invoke)
        result = await wrapped_invoke("Hello")
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, mw: Middleware) -> None:
        """Add middleware to the stack."""
        self._middleware.append(mw)

    def use(self, mw: Middleware) -> Middleware:
        """Decorator to register middleware."""
        self.add(mw)
        return mw

    def wrap(
        self, invoke_fn: Callable[..., Awaitable[Any]], agent_name: str = ""
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap an invoke function with the middleware stack."""

        async def wrapped(message: str, **kwargs: Any) -> Any:
            async def call_chain(msg: str) -> Any:
                return await invoke_fn(msg, **kwargs)

            chain = call_chain
            # Build chain inside-out: reverse so first-added middleware runs outermost
            for mw in reversed(self._middleware):
                prev = chain

                # Closure factory to capture current mw/prev per iteration,
                # avoiding late-binding issues with the loop variable.
                def make_next(
                    m: Middleware, p: Callable[..., Awaitable[Any]]
                ) -> Callable[..., Awaitable[Any]]:
                    async def next_fn(msg: str) -> Any:
                        return await m(agent_name, msg, p)

                    return next_fn

                chain = make_next(mw, prev)

            return await chain(message)

        return wrapped


async def timing_middleware(agent_name: str, message: str, next_fn: InvokeNext) -> Any:
    """Built-in middleware that tracks invocation duration."""
    start = time.monotonic()
    result = await next_fn(message)
    _ = time.monotonic() - start  # duration available for extension
    return result


async def logging_middleware(agent_name: str, message: str, next_fn: InvokeNext) -> Any:
    """Built-in middleware that logs invocations via structured logger."""
    from sktk.observability.logging import get_logger

    logger = get_logger("sktk.middleware")
    logger.info("invoking agent", agent_name=agent_name, input_length=len(message))
    try:
        result = await next_fn(message)
        logger.info("agent completed", agent_name=agent_name)
        return result
    except Exception as e:
        logger.error("agent failed", agent_name=agent_name, exc_info=e)
        raise
