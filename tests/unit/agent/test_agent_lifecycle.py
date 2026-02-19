# tests/unit/agent/test_agent_lifecycle.py
import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.hooks import LifecycleHooks
from sktk.agent.providers import CompletionResult
from sktk.agent.tools import tool
from sktk.core.events import CompletionEvent, MessageEvent, ThinkingEvent, ToolCallEvent
from sktk.core.types import Allow, TokenUsage
from sktk.testing.mocks import MockKernel


@pytest.mark.asyncio
async def test_hooks_fire_on_invoke():
    log = []

    async def on_start(name, inp):
        log.append(("start", name))

    async def on_complete(name, inp, out):
        log.append(("complete", name, out))

    hooks = LifecycleHooks(on_start=[on_start], on_complete=[on_complete])
    agent = SKTKAgent(name="a", instructions="test", hooks=hooks)
    kernel = MockKernel()
    kernel.expect_chat_completion(["hello"])
    agent.kernel = kernel
    result = await agent.invoke("hi")
    assert result == "hello"
    assert log == [("start", "a"), ("complete", "a", "hello")]


@pytest.mark.asyncio
async def test_hooks_fire_on_error():
    log = []

    async def on_start(name, inp):
        log.append("start")

    async def on_error(name, inp, err):
        log.append(("error", str(err)))

    hooks = LifecycleHooks(on_start=[on_start], on_error=[on_error])
    agent = SKTKAgent(name="a", instructions="test", hooks=hooks)
    # No kernel set, will raise NotImplementedError
    with pytest.raises(NotImplementedError):
        await agent.invoke("hi")
    assert log[0] == "start"
    assert log[1][0] == "error"


@pytest.mark.asyncio
async def test_context_manager():
    agent = SKTKAgent(name="a", instructions="test")
    async with agent as a:
        assert a.name == "a"


@pytest.mark.asyncio
async def test_context_manager_exit_closes_session():
    class _SessionWithClose:
        def __init__(self) -> None:
            self.closed = False

        async def close(self):
            self.closed = True

    session = _SessionWithClose()
    agent = SKTKAgent(name="a", instructions="test", session=session)

    await agent.__aexit__(None, None, None)

    assert session.closed is True


@pytest.mark.asyncio
async def test_streaming():
    agent = SKTKAgent(name="a", instructions="test")
    kernel = MockKernel()
    kernel.expect_chat_completion(["hello world"])
    agent.kernel = kernel
    chunks = []
    async for chunk in agent.invoke_stream("hi"):
        chunks.append(chunk)
    assert "".join(chunks).strip() == "hello world"


@pytest.mark.asyncio
async def test_streaming_runs_output_filters():
    calls = []

    class RecordingFilter:
        async def on_input(self, context):
            calls.append(("input", context.content))
            return Allow()

        async def on_output(self, context):
            calls.append(("output", context.content))
            return Allow()

        async def on_function_call(self, context):
            return Allow()

    agent = SKTKAgent(
        name="a",
        instructions="test",
        filters=[RecordingFilter()],
    )
    kernel = MockKernel()
    kernel.expect_chat_completion(["hello world"])
    agent.kernel = kernel

    chunks = []
    async for chunk in agent.invoke_stream("hi"):
        chunks.append(chunk)

    assert any(stage == "output" for stage, _ in calls)
    assert "".join(chunks).strip() == "hello world"


@pytest.mark.asyncio
async def test_streaming_hooks_fire_on_error():
    log = []

    async def on_error(name, inp, err):
        log.append(("error", str(err)))

    agent = SKTKAgent(
        name="a",
        instructions="test",
        hooks=LifecycleHooks(on_error=[on_error]),
    )

    with pytest.raises(NotImplementedError):
        async for _ in agent.invoke_stream("hi"):
            pass

    assert log and log[0][0] == "error"


@pytest.mark.asyncio
async def test_get_tool():
    @tool(description="Add numbers")
    def add(a: int, b: int) -> int:
        return a + b

    agent = SKTKAgent(name="a", instructions="test", tools=[add])
    assert agent.get_tool("add") is not None
    assert agent.get_tool("missing") is None


@pytest.mark.asyncio
async def test_call_tool():
    @tool(description="Add numbers")
    def add(a: int, b: int) -> int:
        return a + b

    agent = SKTKAgent(name="a", instructions="test", tools=[add])
    result = await agent.call_tool("add", a=2, b=3)
    assert result == 5


@pytest.mark.asyncio
async def test_call_tool_missing():
    agent = SKTKAgent(name="a", instructions="test")
    with pytest.raises(KeyError, match="not registered"):
        await agent.call_tool("missing")


class _UsageProvider:
    @property
    def name(self) -> str:
        return "usage-provider"

    async def complete(self, messages, **kwargs):
        return CompletionResult(
            text="live",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=3),
            metadata={"model": "demo"},
        )


@pytest.mark.asyncio
async def test_invoke_emits_thinking_message_completion_events():
    agent = SKTKAgent(name="a", instructions="test", service=_UsageProvider())
    result = await agent.invoke("hi")
    assert result == "live"

    events = list(agent.event_stream.events)
    assert len(events) == 3
    assert isinstance(events[0], ThinkingEvent)
    assert isinstance(events[1], MessageEvent)
    assert isinstance(events[2], CompletionEvent)
    assert events[1].provider == "usage-provider"
    assert events[1].token_usage is not None
    assert events[1].token_usage.total_tokens == 5
    assert events[2].total_tokens is not None
    assert events[2].total_tokens.total_tokens == 5


@pytest.mark.asyncio
async def test_invoke_stream_emits_same_usage_events():
    agent = SKTKAgent(name="a", instructions="test", service=_UsageProvider())
    chunks = []
    async for chunk in agent.invoke_stream("hi"):
        chunks.append(chunk)
    assert "".join(chunks).strip() == "live"

    events = list(agent.event_stream.events)
    assert len(events) == 3
    assert isinstance(events[0], ThinkingEvent)
    assert isinstance(events[1], MessageEvent)
    assert isinstance(events[2], CompletionEvent)
    assert events[1].token_usage is not None
    assert events[1].token_usage.total_tokens == 5
    assert events[2].total_tokens is not None
    assert events[2].total_tokens.total_tokens == 5


@pytest.mark.asyncio
async def test_call_tool_emits_tool_call_event():
    @tool(description="Add numbers")
    def add(a: int, b: int) -> int:
        return a + b

    agent = SKTKAgent(name="a", instructions="test", tools=[add])
    result = await agent.call_tool("add", a=4, b=5)
    assert result == 9

    events = list(agent.event_stream.events)
    assert len(events) == 1
    assert isinstance(events[0], ToolCallEvent)
    assert events[0].function == "add"
    assert events[0].arguments == {"a": 4, "b": 5}
