import asyncio

import pytest

from sktk.core.context import (
    ExecutionContext,
    _current_context,
    context_scope,
    get_context,
    propagate_context,
    require_context,
    set_context,
)
from sktk.core.errors import SKTKContextError


def test_execution_context_creation():
    ctx = ExecutionContext(
        correlation_id="corr-1", tenant_id="tenant-a", user_id="user-1", session_id="sess-1"
    )
    assert ctx.correlation_id == "corr-1"
    assert ctx.tenant_id == "tenant-a"
    assert ctx.metadata == {}


def test_execution_context_is_frozen():
    ctx = ExecutionContext(correlation_id="corr-1")
    with pytest.raises(AttributeError):
        ctx.correlation_id = "changed"


def test_get_context_returns_none_by_default():
    assert get_context() is None


def test_require_context_raises_without_context():
    with pytest.raises(SKTKContextError, match="No execution context"):
        require_context()


def test_set_and_get_context():
    ctx = ExecutionContext(correlation_id="corr-1")
    token = set_context(ctx)
    try:
        assert get_context() is ctx
        assert require_context() is ctx
    finally:
        _current_context.reset(token)


@pytest.mark.asyncio
async def test_context_scope_sets_and_restores():
    assert get_context() is None
    ctx = ExecutionContext(correlation_id="scoped-1")
    async with context_scope(ctx):
        assert get_context() is ctx
    assert get_context() is None


@pytest.mark.asyncio
async def test_context_scope_restores_previous():
    outer = ExecutionContext(correlation_id="outer")
    inner = ExecutionContext(correlation_id="inner")
    token = set_context(outer)
    try:
        async with context_scope(inner):
            assert get_context() is inner
        assert get_context() is outer
    finally:
        _current_context.reset(token)


@pytest.mark.asyncio
async def test_context_scope_auto_generates_correlation_id():
    async with context_scope() as ctx:
        assert ctx.correlation_id is not None
        assert len(ctx.correlation_id) > 0
        assert get_context() is ctx
    assert get_context() is None


@pytest.mark.asyncio
async def test_context_scope_restores_on_exception():
    ctx = ExecutionContext(correlation_id="will-fail")
    with pytest.raises(ValueError, match="boom"):
        async with context_scope(ctx):
            assert get_context() is ctx
            raise ValueError("boom")
    assert get_context() is None


@pytest.mark.asyncio
async def test_propagate_context_to_task():
    ctx = ExecutionContext(correlation_id="propagated")
    result = []

    @propagate_context
    async def worker():
        result.append(get_context())

    async with context_scope(ctx):
        task = asyncio.create_task(worker())
        await task

    assert result[0] is ctx


@pytest.mark.asyncio
async def test_propagate_context_nested_tasks():
    ctx = ExecutionContext(correlation_id="parent")
    results = []

    @propagate_context
    async def child():
        results.append(get_context())

    @propagate_context
    async def parent():
        results.append(get_context())
        task = asyncio.create_task(child())
        await task

    async with context_scope(ctx):
        task = asyncio.create_task(parent())
        await task

    assert all(r is ctx for r in results)


@pytest.mark.asyncio
async def test_context_scope_unexpected_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        async with context_scope(foo="bar"):
            pass


@pytest.mark.asyncio
async def test_propagate_context_without_active_context():
    """propagate_context works even when no context is set."""
    result = []

    @propagate_context
    async def worker():
        result.append(get_context())
        return "done"

    assert get_context() is None
    r = await worker()
    assert r == "done"
    assert result[0] is None
