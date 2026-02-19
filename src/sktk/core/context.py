"""Execution context propagation via contextvars."""

from __future__ import annotations

import functools
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, AsyncIterator, Awaitable, Callable, ParamSpec, TypeVar

from sktk.core.errors import SKTKContextError

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context carried through async call chains."""

    correlation_id: str
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    parent_correlation_id: str | None = None
    metadata: MappingProxyType[str, str] = field(default_factory=lambda: MappingProxyType({}))


_current_context: ContextVar[ExecutionContext | None] = ContextVar(
    "skat_execution_context", default=None
)


def get_context() -> ExecutionContext | None:
    """Return the current execution context, or None if not set."""
    return _current_context.get()


def require_context() -> ExecutionContext:
    """Return the current execution context, raising if not set."""
    ctx = _current_context.get()
    if ctx is None:
        raise SKTKContextError("No execution context set. Use context_scope() or set_context().")
    return ctx


def set_context(ctx: ExecutionContext) -> Token[ExecutionContext | None]:
    """Set execution context, returning a token for reset."""
    return _current_context.set(ctx)


@asynccontextmanager
async def context_scope(
    ctx: ExecutionContext | None = None, **kwargs: Any
) -> AsyncIterator[ExecutionContext]:
    """Async context manager that sets execution context and restores on exit."""
    if ctx is None:
        metadata = kwargs.pop("metadata", {})
        ctx = ExecutionContext(
            correlation_id=kwargs.pop("correlation_id", str(uuid.uuid4())),
            tenant_id=kwargs.pop("tenant_id", None),
            user_id=kwargs.pop("user_id", None),
            session_id=kwargs.pop("session_id", None),
            parent_correlation_id=kwargs.pop("parent_correlation_id", None),
            metadata=MappingProxyType(metadata if isinstance(metadata, dict) else {}),
        )
        if kwargs:
            valid_params = [
                "correlation_id",
                "tenant_id",
                "user_id",
                "session_id",
                "parent_correlation_id",
                "metadata",
            ]
            raise TypeError(
                f"context_scope() got unexpected keyword arguments: {list(kwargs)}. "
                f"Valid parameters: {valid_params}"
            )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


def propagate_context(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator that captures the current execution context at call time
    and restores it inside the wrapped async coroutine. Useful when scheduling
    work via asyncio.create_task. Does NOT propagate context across threads
    (e.g. loop.run_in_executor), since contextvars are thread-local."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Captured at call-site so the value is available even if the
        # coroutine runs in a different context (e.g. a new task/thread).
        captured = _current_context.get()
        if captured is not None:
            token = _current_context.set(captured)
            try:
                return await fn(*args, **kwargs)
            finally:
                _current_context.reset(token)
        return await fn(*args, **kwargs)

    return wrapper
