# tests/unit/agent/test_agent.py
import asyncio

import pytest
from pydantic import BaseModel

from sktk.agent.agent import SKTKAgent
from sktk.agent.capabilities import Capability
from sktk.agent.providers import CompletionResult, ToolCallRequest
from sktk.agent.router import FallbackPolicy, Router
from sktk.agent.tools import Tool
from sktk.core.types import TokenUsage, maybe_await
from sktk.session.session import Session
from sktk.testing.mocks import MockKernel


class Request(BaseModel):
    query: str


class Response(BaseModel):
    answer: str


def test_agent_creation_minimal():
    agent = SKTKAgent(name="test-agent", instructions="You are a test agent.")
    assert agent.name == "test-agent"
    assert agent.instructions == "You are a test agent."


def test_agent_creation_full():
    session = Session(id="s1")
    agent = SKTKAgent(
        name="analyst",
        instructions="Analyze data.",
        session=session,
        capabilities=[
            Capability(
                name="analysis",
                description="Analyze",
                input_types=[Request],
                output_types=[Response],
            )
        ],
        input_contract=Request,
        output_contract=Response,
        max_iterations=5,
        timeout=30.0,
    )
    assert agent.session is session
    assert len(agent.capabilities) == 1
    assert agent.input_contract is Request
    assert agent.output_contract is Response
    assert agent.max_iterations == 5
    assert agent.timeout == 30.0


def test_agent_default_values():
    agent = SKTKAgent(name="basic", instructions="Do stuff.")
    assert agent.session is None
    assert agent.capabilities == []
    assert agent.input_contract is None
    assert agent.output_contract is None
    assert agent.max_iterations == 10
    assert agent.timeout == 60.0
    assert agent.filters == []


@pytest.mark.asyncio
async def test_agent_invoke_string_input():
    agent = SKTKAgent(name="test", instructions="You help.")
    mk = MockKernel()
    mk.expect_chat_completion(responses=["Hello back!"])
    agent.kernel = mk
    result = await agent.invoke("Hello")
    assert result == "Hello back!"


@pytest.mark.asyncio
async def test_agent_invoke_with_output_contract():
    agent = SKTKAgent(name="test", instructions="Return JSON.", output_contract=Response)
    mk = MockKernel()
    mk.expect_chat_completion(responses=['{"answer": "42"}'])
    agent.kernel = mk
    result = await agent.invoke("What is the answer?")
    assert isinstance(result, Response)
    assert result.answer == "42"


@pytest.mark.asyncio
async def test_agent_invoke_with_input_contract():
    agent = SKTKAgent(name="test", instructions="Process request.", input_contract=Request)
    mk = MockKernel()
    mk.expect_chat_completion(responses=["processed"])
    agent.kernel = mk
    result = await agent.invoke(Request(query="test query"))
    assert result == "processed"


@pytest.mark.asyncio
async def test_agent_invoke_persists_to_session():
    session = Session(id="s1")
    agent = SKTKAgent(name="test", instructions="Help.", session=session)
    mk = MockKernel()
    mk.expect_chat_completion(responses=["response"])
    agent.kernel = mk
    await agent.invoke("question")
    messages = await session.history.get()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "question"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "response"


@pytest.mark.asyncio
async def test_agent_invoke_filter_blocks_input():
    from sktk.agent.filters import ContentSafetyFilter
    from sktk.core.errors import GuardrailException

    agent = SKTKAgent(
        name="test",
        instructions="Help.",
        filters=[ContentSafetyFilter(blocked_patterns=[r"forbidden"])],
    )
    mk = MockKernel()
    mk.expect_chat_completion(responses=["should not reach"])
    agent.kernel = mk
    with pytest.raises(GuardrailException, match="forbidden"):
        await agent.invoke("this is forbidden content")


@pytest.mark.asyncio
async def test_agent_invoke_filter_blocks_output():
    from sktk.agent.filters import PIIFilter
    from sktk.core.errors import GuardrailException

    agent = SKTKAgent(
        name="test",
        instructions="Help.",
        filters=[PIIFilter()],
    )
    mk = MockKernel()
    mk.expect_chat_completion(responses=["Contact me at test@example.com"])
    agent.kernel = mk
    with pytest.raises(GuardrailException):
        await agent.invoke("Show contacts")


def test_agent_rshift_not_implemented():
    agent = SKTKAgent(name="a", instructions="A.")
    result = agent.__rshift__(42)
    assert result is NotImplemented


def test_agent_rshift_to_node():
    from sktk.team.topology import AgentNode, SequentialNode

    a = SKTKAgent(name="a", instructions="A.")
    b = SKTKAgent(name="b", instructions="B.")
    node = AgentNode(agent=b)
    result = a >> node
    assert isinstance(result, SequentialNode)


@pytest.mark.asyncio
async def test_agent_invoke_without_kernel():
    agent = SKTKAgent(name="test", instructions="Help.")
    with pytest.raises(NotImplementedError, match="kernel"):
        await agent.invoke("hello")


@pytest.mark.asyncio
async def test_agent_with_responses():
    agent = SKTKAgent.with_responses("bot", ["Hello!", "Goodbye!"])
    assert agent.name == "bot"
    assert agent.instructions == ""
    assert await agent.invoke("hi") == "Hello!"
    assert await agent.invoke("bye") == "Goodbye!"


@pytest.mark.asyncio
async def test_agent_with_responses_kwargs():
    agent = SKTKAgent.with_responses(
        "bot",
        ["ok"],
        instructions="Be helpful.",
        max_iterations=3,
    )
    assert agent.instructions == "Be helpful."
    assert agent.max_iterations == 3
    assert await agent.invoke("test") == "ok"


@pytest.mark.asyncio
async def test_agent_call_tool_runs_function_filters():
    from sktk.agent.filters import ContentSafetyFilter
    from sktk.agent.tools import tool
    from sktk.core.errors import GuardrailException

    @tool(description="Safe tool")
    def safe():
        return "ok"

    @tool(name="danger", description="Dangerous tool")
    def danger():
        return "boom"

    agent = SKTKAgent(
        name="test",
        instructions="Hi",
        filters=[ContentSafetyFilter(blocked_patterns=[r"danger"])],
        tools=[safe, danger],
    )

    with pytest.raises(GuardrailException):
        await agent.call_tool("danger")

    assert await agent.call_tool("safe") == "ok"


@pytest.mark.asyncio
async def test_agent_invoke_applies_input_modify_filter():
    from sktk.agent.filters import FilterContext
    from sktk.core.types import Allow, Modify

    class PrefixInputFilter:
        async def on_input(self, context: FilterContext):
            return Modify(content=f"[prefix]{context.content}")

        async def on_output(self, context: FilterContext):
            return Allow()

        async def on_function_call(self, context: FilterContext):
            return Allow()

    class EchoAgent(SKTKAgent):
        async def _get_response(self, prompt: str, **kwargs):
            return prompt

    agent = EchoAgent(name="echo", instructions="Echo.", filters=[PrefixInputFilter()])
    result = await agent.invoke("hello")
    assert result == "[prefix]hello"


@pytest.mark.asyncio
async def test_agent_invoke_applies_output_modify_filter():
    from sktk.agent.filters import FilterContext
    from sktk.core.types import Allow, Modify

    class UppercaseOutputFilter:
        async def on_input(self, context: FilterContext):
            return Allow()

        async def on_output(self, context: FilterContext):
            return Modify(content=context.content.upper())

        async def on_function_call(self, context: FilterContext):
            return Allow()

    agent = SKTKAgent(name="test", instructions="Help.", filters=[UppercaseOutputFilter()])
    mk = MockKernel()
    mk.expect_chat_completion(responses=["lowercase"])
    agent.kernel = mk

    result = await agent.invoke("hello")
    assert result == "LOWERCASE"


class _Provider:
    @property
    def name(self) -> str:
        return "provider-a"

    async def complete(self, messages, **kwargs):
        return CompletionResult(
            text="live response",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=4),
            metadata={"model": "demo-a"},
        )


class _SlowProvider:
    @property
    def name(self) -> str:
        return "slow"

    async def complete(self, messages, **kwargs):
        import asyncio

        await asyncio.sleep(0.05)
        return "too slow"


class _StreamingService:
    async def stream_with_metadata(self, messages, **kwargs):
        async def _iter():
            yield "hello "
            yield "world"

        return _iter(), {
            "provider": "stream-provider",
            "usage": TokenUsage(prompt_tokens=5, completion_tokens=6),
        }


@pytest.mark.asyncio
async def test_agent_invokes_service_router_and_captures_metadata():
    router = Router([_Provider()], policy=FallbackPolicy())
    agent = SKTKAgent(name="live", instructions="Use provider", service=router)

    result = await agent.invoke("hello")

    assert result == "live response"
    assert agent._last_provider == "provider-a"
    assert agent._last_usage is not None
    assert agent._last_usage.total_tokens == 7
    assert agent._last_response_metadata["model"] == "demo-a"


@pytest.mark.asyncio
async def test_agent_timeout_applies_to_service_calls():
    router = Router([_SlowProvider()], policy=FallbackPolicy())
    agent = SKTKAgent(name="live", instructions="Use provider", service=router, timeout=0.01)

    with pytest.raises(TimeoutError):
        await agent.invoke("hello")


@pytest.mark.asyncio
async def test_agent_stream_with_metadata_parity():
    agent = SKTKAgent(name="live", instructions="Use provider", service=_StreamingService())

    chunks = []
    async for chunk in agent.invoke_stream("hello"):
        chunks.append(chunk)

    assert "".join(chunks) == "hello world"
    assert agent._last_provider == "stream-provider"
    assert agent._last_usage is not None
    assert agent._last_usage.total_tokens == 11


def test_coerce_stream_chunk_tuple_metadata():
    agent = SKTKAgent(name="helper", instructions="Helpers.")
    text, metadata = agent._runtime.coerce_stream_chunk(("chunk", {"provider": "demo"}))
    assert text == "chunk"
    assert metadata == {"provider": "demo"}


def test_unpack_stream_result_non_iterator_raises():
    agent = SKTKAgent(name="helper", instructions="Helpers.")
    with pytest.raises(TypeError, match="stream_with_metadata must return"):
        agent._runtime.unpack_stream_result(("not-an-iterator", {"provider": "demo"}))


def test_resolve_timeout_from_kwargs_pops_timeout():
    agent = SKTKAgent(name="helper", instructions="Helpers.", timeout=60.0)
    kwargs = {"timeout": "2.5", "temperature": 0.1}
    timeout = agent._runtime.resolve_timeout(kwargs)
    assert timeout == pytest.approx(2.5)
    assert "timeout" not in kwargs
    assert kwargs["temperature"] == 0.1


def test_resolve_timeout_none_returns_none():
    agent = SKTKAgent(name="helper", instructions="Helpers.", timeout=60.0)
    assert agent._runtime.resolve_timeout({"timeout": None}) is None

    no_default_timeout = SKTKAgent(name="helper2", instructions="Helpers.", timeout=None)
    assert no_default_timeout._runtime.resolve_timeout({}) is None


@pytest.mark.parametrize("timeout", [0, -1, -0.1])
def test_resolve_timeout_rejects_non_positive(timeout):
    agent = SKTKAgent(name="helper", instructions="Helpers.", timeout=60.0)
    with pytest.raises(TimeoutError, match="timeout must be > 0"):
        agent._runtime.resolve_timeout({"timeout": timeout})


def test_record_response_metadata_parses_usage_dict():
    agent = SKTKAgent(name="helper", instructions="Helpers.")
    agent._runtime.record_response_metadata(
        {
            "provider": "dict-provider",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_cost_usd": 0.42,
            },
        }
    )
    assert agent._last_provider == "dict-provider"
    assert agent._last_usage is not None
    assert agent._last_usage.prompt_tokens == 3
    assert agent._last_usage.completion_tokens == 4
    assert agent._last_usage.total_cost_usd == pytest.approx(0.42)


def test_coerce_text_supports_content_text_and_fallback():
    with_content = type("WithContent", (), {"content": "content-value"})()
    with_text = type("WithText", (), {"text": "text-value"})()

    class CustomObject:
        def __str__(self) -> str:
            return "object-value"

    assert SKTKAgent._coerce_text("plain") == "plain"
    assert SKTKAgent._coerce_text(with_content) == "content-value"
    assert SKTKAgent._coerce_text(with_text) == "text-value"
    assert SKTKAgent._coerce_text(CustomObject()) == "object-value"


@pytest.mark.asyncio
async def testmaybe_await_returns_non_awaitable():
    marker = object()
    assert await maybe_await(marker) is marker


@pytest.mark.asyncio
async def test_iterate_stream_with_timeout_none_yields_all_chunks():
    agent = SKTKAgent(name="stream", instructions="Helpers.")

    async def stream():
        yield "a"
        yield "b"

    chunks = []
    async for chunk in agent._runtime.iterate_stream_with_timeout(stream(), timeout=None):
        chunks.append(chunk)
    assert chunks == ["a", "b"]


@pytest.mark.asyncio
async def test_iterate_stream_with_timeout_yields_before_deadline():
    agent = SKTKAgent(name="stream", instructions="Helpers.")

    async def stream():
        await asyncio.sleep(0.001)
        yield "a"
        await asyncio.sleep(0.001)
        yield "b"

    chunks = []
    async for chunk in agent._runtime.iterate_stream_with_timeout(stream(), timeout=0.2):
        chunks.append(chunk)
    assert chunks == ["a", "b"]


@pytest.mark.asyncio
async def test_iterate_stream_with_timeout_raises_when_deadline_exceeded():
    agent = SKTKAgent(name="stream", instructions="Helpers.")

    async def slow_stream():
        await asyncio.sleep(0.05)
        yield "late"

    with pytest.raises(TimeoutError, match="timed out"):
        async for _ in agent._runtime.iterate_stream_with_timeout(slow_stream(), timeout=0.01):
            pass


@pytest.mark.asyncio
async def test_invoke_stream_applies_base_model_and_modify_filters_and_session_history():
    from sktk.agent.filters import FilterContext
    from sktk.core.types import Allow, Modify

    class StreamModifyFilter:
        async def on_input(self, context: FilterContext):
            return Modify(content="rewritten-prompt")

        async def on_output(self, context: FilterContext):
            return Modify(content="rewritten-output")

        async def on_function_call(self, context: FilterContext):
            return Allow()

    session = Session(id="stream-session")
    agent = SKTKAgent(
        name="stream",
        instructions="Helpers.",
        filters=[StreamModifyFilter()],
        session=session,
    )
    kernel = MockKernel()
    kernel.expect_chat_completion(["hello world"])
    agent.kernel = kernel

    chunks = []
    async for chunk in agent.invoke_stream(Request(query="original")):
        chunks.append(chunk)

    # Chunks are streamed in real-time; output filter modifies the accumulated text
    # used for session history, but the consumer sees raw chunks.
    assert "".join(chunks).strip() == "hello world"
    messages = await session.history.get()
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "rewritten-prompt"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "rewritten-output"


@pytest.mark.asyncio
async def test_invoke_stream_input_filter_deny_raises_guardrail_exception():
    from sktk.agent.filters import FilterContext
    from sktk.core.errors import GuardrailException
    from sktk.core.types import Allow, Deny

    class DenyInputFilter:
        async def on_input(self, context: FilterContext):
            return Deny(reason="blocked input")

        async def on_output(self, context: FilterContext):
            return Allow()

        async def on_function_call(self, context: FilterContext):
            return Allow()

    agent = SKTKAgent(name="stream", instructions="Helpers.", filters=[DenyInputFilter()])
    kernel = MockKernel()
    kernel.expect_chat_completion(["unused"])
    agent.kernel = kernel

    with pytest.raises(GuardrailException, match="blocked input"):
        async for _ in agent.invoke_stream("hello"):
            pass


@pytest.mark.asyncio
async def test_invoke_stream_output_filter_deny_raises_guardrail_exception():
    from sktk.agent.filters import FilterContext
    from sktk.core.errors import GuardrailException
    from sktk.core.types import Allow, Deny

    class DenyOutputFilter:
        async def on_input(self, context: FilterContext):
            return Allow()

        async def on_output(self, context: FilterContext):
            return Deny(reason="blocked output")

        async def on_function_call(self, context: FilterContext):
            return Allow()

    agent = SKTKAgent(name="stream", instructions="Helpers.", filters=[DenyOutputFilter()])
    kernel = MockKernel()
    kernel.expect_chat_completion(["hello world"])
    agent.kernel = kernel

    with pytest.raises(GuardrailException, match="blocked output"):
        async for _ in agent.invoke_stream("hello"):
            pass


@pytest.mark.asyncio
async def test_get_response_uses_sk_agent_invoke_and_coerces_text():
    class FakeSKAgent:
        async def invoke(self, prompt: str, **kwargs):
            return type("Response", (), {"content": f"from-sk:{prompt}"})()

    agent = SKTKAgent(name="sk", instructions="Helpers.", timeout=1.0, sk_agent=FakeSKAgent())
    result = await agent._get_response("prompt")
    assert result == "from-sk:prompt"


@pytest.mark.asyncio
async def test_get_response_stream_stream_with_metadata_merges_chunk_metadata():
    class ServiceWithMetadataStream:
        async def stream_with_metadata(self, messages, **kwargs):
            async def _stream():
                yield ("a", {"provider": "chunk-provider"})
                yield ("b", {"usage": {"prompt_tokens": 2, "completion_tokens": 3}})

            return _stream(), {"provider": "initial-provider"}

    agent = SKTKAgent(
        name="stream-meta", instructions="Helpers.", service=ServiceWithMetadataStream()
    )
    chunks = [chunk async for chunk in agent._get_response_stream("hello")]

    assert chunks == ["a", "b"]
    assert agent._last_provider == "chunk-provider"
    assert agent._last_usage is not None
    assert agent._last_usage.total_tokens == 5


@pytest.mark.asyncio
async def test_get_response_stream_stream_path_records_chunk_metadata():
    class ServiceWithStreamOnly:
        def stream(self, messages, **kwargs):
            async def _stream():
                yield ("x", {"provider": "stream-provider"})
                yield ("y", {"usage": {"prompt_tokens": 1, "completion_tokens": 4}})

            return _stream()

    agent = SKTKAgent(name="stream-only", instructions="Helpers.", service=ServiceWithStreamOnly())
    chunks = [chunk async for chunk in agent._get_response_stream("hello")]

    assert chunks == ["x", "y"]
    assert agent._last_provider == "stream-provider"
    assert agent._last_usage is not None
    assert agent._last_usage.total_tokens == 5


@pytest.mark.asyncio
async def test_unpack_stream_result_accepts_async_iterator_without_metadata_tuple():
    agent = SKTKAgent(name="helper", instructions="Helpers.")

    async def stream():
        yield "chunk"

    unpacked_stream, metadata = agent._runtime.unpack_stream_result(stream())
    assert metadata == {}
    assert [chunk async for chunk in unpacked_stream] == ["chunk"]


def test_coerce_stream_chunk_completion_result_includes_usage_metadata():
    agent = SKTKAgent(name="helper", instructions="Helpers.")
    usage = TokenUsage(prompt_tokens=4, completion_tokens=1)
    text, metadata = agent._runtime.coerce_stream_chunk(
        CompletionResult(text="done", usage=usage, metadata={"provider": "demo"})
    )
    assert text == "done"
    assert metadata["provider"] == "demo"
    assert metadata["usage"] == usage


@pytest.mark.asyncio
async def test_await_with_timeout_none_awaits_call_without_wait_for():
    agent = SKTKAgent(name="helper", instructions="Helpers.")

    async def immediate():
        return "ok"

    assert await agent._runtime.await_with_timeout(immediate(), timeout=None) == "ok"


@pytest.mark.asyncio
async def test_iterate_stream_with_timeout_zero_raises_immediately():
    agent = SKTKAgent(name="helper", instructions="Helpers.")

    async def stream():
        yield "never"

    with pytest.raises(TimeoutError, match="timed out"):
        async for _ in agent._runtime.iterate_stream_with_timeout(stream(), timeout=0.0):
            pass


class _NeverCalledProvider:
    """Service whose complete() should never be reached."""

    def __init__(self):
        self.call_count = 0

    async def complete(self, messages, **kwargs):
        self.call_count += 1
        return CompletionResult(text="should not reach")


@pytest.mark.asyncio
async def test_max_iterations_zero_returns_empty_via_service():
    provider = _NeverCalledProvider()
    agent = SKTKAgent(
        name="zero-iter",
        instructions="Help.",
        service=provider,
        max_iterations=0,
    )
    result = await agent.invoke("hello")
    assert result == ""
    assert provider.call_count == 0


# ---------------------------------------------------------------
# Tool-call loop tests
# ---------------------------------------------------------------


class _ScriptedProvider:
    """Provider returning a pre-configured sequence of CompletionResults."""

    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.received_messages: list[list[dict]] = []

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, messages, **kwargs):
        self.received_messages.append(list(messages))
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


@pytest.mark.asyncio
async def test_tool_call_loop_single_turn_text_only():
    """LLM returns text without tool calls -- returned directly."""
    provider = _ScriptedProvider(
        [
            CompletionResult(text="Hello there"),
        ]
    )
    agent = SKTKAgent(name="t", instructions="Help.", service=provider)
    result = await agent.invoke("hi")
    assert result == "Hello there"


@pytest.mark.asyncio
async def test_tool_call_loop_multi_turn_tool_then_text():
    """LLM calls a tool, gets result, then returns final text."""

    async def add(a: int, b: int) -> str:
        return str(a + b)

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        fn=add,
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )

    provider = _ScriptedProvider(
        [
            CompletionResult(
                text="",
                tool_calls=[ToolCallRequest(id="tc1", name="add", arguments={"a": 2, "b": 3})],
            ),
            CompletionResult(text="The sum is 5"),
        ]
    )
    agent = SKTKAgent(name="t", instructions="Help.", service=provider, tools=[add_tool])
    result = await agent.invoke("add 2+3")
    assert result == "The sum is 5"
    # Verify the tool result was passed back to LLM
    second_call_msgs = provider.received_messages[1]
    tool_msg = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert tool_msg[0]["content"] == "5"
    assert tool_msg[0]["tool_call_id"] == "tc1"


@pytest.mark.asyncio
async def test_tool_call_loop_unknown_tool_sends_error_to_llm():
    """LLM calls a tool that doesn't exist -- error message sent back."""
    provider = _ScriptedProvider(
        [
            CompletionResult(
                text="",
                tool_calls=[ToolCallRequest(id="tc1", name="nonexistent", arguments={})],
            ),
            CompletionResult(text="Sorry, tool not found"),
        ]
    )
    agent = SKTKAgent(name="t", instructions="Help.", service=provider)
    result = await agent.invoke("call nonexistent")
    assert result == "Sorry, tool not found"
    # Verify error message was fed back
    second_call_msgs = provider.received_messages[1]
    tool_msg = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert "not found" in tool_msg[0]["content"]


@pytest.mark.asyncio
async def test_tool_call_loop_validation_error_sends_error_to_llm():
    """LLM passes wrong args -- ValueError sent back to LLM."""

    async def strict_tool(query: str) -> str:
        return f"result for {query}"

    strict = Tool(
        name="strict",
        description="Requires query",
        fn=strict_tool,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    provider = _ScriptedProvider(
        [
            CompletionResult(
                text="",
                tool_calls=[ToolCallRequest(id="tc1", name="strict", arguments={})],
            ),
            CompletionResult(text="Fixed it"),
        ]
    )
    agent = SKTKAgent(name="t", instructions="Help.", service=provider, tools=[strict])
    result = await agent.invoke("do something")
    assert result == "Fixed it"
    # Verify validation error was sent back
    second_call_msgs = provider.received_messages[1]
    tool_msg = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert "Error:" in tool_msg[0]["content"]
    assert "missing required" in tool_msg[0]["content"].lower()


@pytest.mark.asyncio
async def test_tool_call_loop_max_iterations_exceeded():
    """LLM keeps calling tools -- returns last text after max iterations."""
    call_count = 0

    async def echo(**kwargs) -> str:
        nonlocal call_count
        call_count += 1
        return f"echo-{call_count}"

    echo_tool = Tool(name="echo", description="Echo", fn=echo, parameters={})

    # LLM always returns tool calls, never plain text
    infinite_tool_calls = [
        CompletionResult(
            text=f"still thinking {i}",
            tool_calls=[ToolCallRequest(id=f"tc{i}", name="echo", arguments={})],
        )
        for i in range(5)
    ]
    provider = _ScriptedProvider(infinite_tool_calls)
    agent = SKTKAgent(
        name="t",
        instructions="Help.",
        service=provider,
        tools=[echo_tool],
        max_iterations=3,
    )
    result = await agent.invoke("loop forever")
    # Should stop after max_iterations and return last text
    assert "still thinking" in result
    assert call_count == 3  # tool was called each iteration
