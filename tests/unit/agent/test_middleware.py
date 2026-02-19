# tests/unit/agent/test_middleware.py
import pytest

from sktk.agent.middleware import MiddlewareStack, logging_middleware, timing_middleware


@pytest.mark.asyncio
async def test_middleware_stack_wraps():
    log = []

    async def track(agent_name, message, next_fn):
        log.append(f"before:{message}")
        result = await next_fn(message)
        log.append(f"after:{result}")
        return result

    stack = MiddlewareStack()
    stack.add(track)

    async def invoke(msg):
        return f"reply:{msg}"

    wrapped = stack.wrap(invoke, agent_name="test")
    result = await wrapped("hello")
    assert result == "reply:hello"
    assert log == ["before:hello", "after:reply:hello"]


@pytest.mark.asyncio
async def test_middleware_stack_ordering():
    log = []

    async def mw_a(name, msg, next_fn):
        log.append("a_before")
        result = await next_fn(msg)
        log.append("a_after")
        return result

    async def mw_b(name, msg, next_fn):
        log.append("b_before")
        result = await next_fn(msg)
        log.append("b_after")
        return result

    stack = MiddlewareStack()
    stack.add(mw_a)
    stack.add(mw_b)

    async def invoke(msg):
        return "ok"

    wrapped = stack.wrap(invoke)
    await wrapped("test")
    assert log == ["a_before", "b_before", "b_after", "a_after"]


@pytest.mark.asyncio
async def test_use_decorator():
    stack = MiddlewareStack()

    @stack.use
    async def my_mw(name, msg, next_fn):
        return await next_fn(msg)

    assert len(stack._middleware) == 1


@pytest.mark.asyncio
async def test_timing_middleware():
    async def invoke(msg):
        return "ok"

    async def next_fn(msg):
        return "ok"

    result = await timing_middleware("test", "hello", next_fn)
    assert result == "ok"


@pytest.mark.asyncio
async def test_logging_middleware():
    async def next_fn(msg):
        return "ok"

    result = await logging_middleware("test", "hello", next_fn)
    assert result == "ok"


@pytest.mark.asyncio
async def test_logging_middleware_error():
    async def next_fn(msg):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await logging_middleware("test", "hello", next_fn)


@pytest.mark.asyncio
async def test_logging_middleware_error_passes_exc_info(monkeypatch):
    calls = []

    class FakeLogger:
        def info(self, msg, **kwargs):
            calls.append(("info", msg, kwargs))

        def error(self, msg, **kwargs):
            calls.append(("error", msg, kwargs))

    fake_logger = FakeLogger()
    monkeypatch.setattr("sktk.observability.logging.get_logger", lambda _: fake_logger)

    async def next_fn(msg):
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError):
        await logging_middleware("test", "hello", next_fn)

    error_call = [c for c in calls if c[0] == "error"][0]
    assert error_call[2]["exc_info"].args[0] == "bad"


@pytest.mark.asyncio
async def test_empty_stack():
    stack = MiddlewareStack()

    async def invoke(msg):
        return f"echo:{msg}"

    wrapped = stack.wrap(invoke)
    result = await wrapped("test")
    assert result == "echo:test"


@pytest.mark.asyncio
async def test_error_propagation_through_chain():
    """Errors raised in the inner invoke propagate through all middleware."""
    cleanup_log = []

    async def outer_mw(name, msg, next_fn):
        try:
            return await next_fn(msg)
        finally:
            cleanup_log.append("outer_cleanup")

    async def inner_mw(name, msg, next_fn):
        try:
            return await next_fn(msg)
        finally:
            cleanup_log.append("inner_cleanup")

    stack = MiddlewareStack()
    stack.add(outer_mw)
    stack.add(inner_mw)

    async def failing_invoke(msg):
        raise RuntimeError("invoke failed")

    wrapped = stack.wrap(failing_invoke)
    with pytest.raises(RuntimeError, match="invoke failed"):
        await wrapped("test")

    # Both middleware should have run their cleanup in inner-to-outer order
    assert cleanup_log == ["inner_cleanup", "outer_cleanup"]


@pytest.mark.asyncio
async def test_error_in_middleware_stops_chain():
    """An error raised in middleware itself prevents further chain execution."""
    stack = MiddlewareStack()

    async def failing_mw(name, msg, next_fn):
        raise ValueError("middleware error")

    async def never_reached_mw(name, msg, next_fn):
        raise AssertionError("should not be reached")

    stack.add(failing_mw)
    stack.add(never_reached_mw)

    async def invoke(msg):
        return "ok"

    wrapped = stack.wrap(invoke)
    with pytest.raises(ValueError, match="middleware error"):
        await wrapped("test")


@pytest.mark.asyncio
async def test_wrap_with_multiple_middleware():
    """wrap() correctly composes three middleware with the invoke function."""
    log = []

    async def mw_1(name, msg, next_fn):
        log.append("1_in")
        result = await next_fn(msg)
        log.append("1_out")
        return result

    async def mw_2(name, msg, next_fn):
        log.append("2_in")
        result = await next_fn(msg)
        log.append("2_out")
        return result

    async def mw_3(name, msg, next_fn):
        log.append("3_in")
        result = await next_fn(msg)
        log.append("3_out")
        return result

    stack = MiddlewareStack()
    stack.add(mw_1)
    stack.add(mw_2)
    stack.add(mw_3)

    async def invoke(msg):
        log.append("invoke")
        return "done"

    wrapped = stack.wrap(invoke, agent_name="agent")
    result = await wrapped("go")

    assert result == "done"
    assert log == ["1_in", "2_in", "3_in", "invoke", "3_out", "2_out", "1_out"]


@pytest.mark.asyncio
async def test_middleware_modifies_request():
    """Middleware can modify the message before passing it downstream."""

    async def uppercaser(name, msg, next_fn):
        return await next_fn(msg.upper())

    stack = MiddlewareStack()
    stack.add(uppercaser)

    async def invoke(msg):
        return f"got:{msg}"

    wrapped = stack.wrap(invoke)
    result = await wrapped("hello")
    assert result == "got:HELLO"


@pytest.mark.asyncio
async def test_middleware_modifies_response():
    """Middleware can modify the response returned from downstream."""

    async def wrapper_mw(name, msg, next_fn):
        result = await next_fn(msg)
        return f"[wrapped]{result}[/wrapped]"

    stack = MiddlewareStack()
    stack.add(wrapper_mw)

    async def invoke(msg):
        return f"reply:{msg}"

    wrapped = stack.wrap(invoke)
    result = await wrapped("test")
    assert result == "[wrapped]reply:test[/wrapped]"


@pytest.mark.asyncio
async def test_middleware_modifies_both_request_and_response():
    """Middleware can transform both request and response."""

    async def transform_mw(name, msg, next_fn):
        result = await next_fn(f"prefix-{msg}")
        return result.replace("prefix-", "transformed-")

    stack = MiddlewareStack()
    stack.add(transform_mw)

    async def invoke(msg):
        return f"echo:{msg}"

    wrapped = stack.wrap(invoke)
    result = await wrapped("data")
    assert result == "echo:transformed-data"
