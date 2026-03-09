"""SKTKAgent -- core agent abstraction wrapping SK's ChatCompletionAgent.

Architectural Overview:
-----------------------

Agent Execution Flow:
    1. Input Validation → [Sync]
    2. Pre-processing Filters → [Async]
    3. LLM Service Call → [Async] → Streaming Response
    4. Post-processing Filters → [Async]
    5. Output Persistence → [Async]

Component Layers:
    - Presentation: Agent interface (async)
    - Business Logic: Filters, hooks, contracts (mixed sync/async)
    - Data Access: Providers, session, events (async)

Concurrency Model:
    - I/O operations: Async/Await (network, DB, file I/O)
    - CPU operations: ThreadPoolExecutor (embeddings, validation)
    - Synchronization: asyncio.Lock for async contexts

Error Handling Strategy:
    - Validation errors: Recoverable, logged at DEBUG level
    - Timeout errors: Recoverable, logged at WARNING level
    - Network errors: Retryable, logged at ERROR level
    - Unexpected errors: Fatal, logged at CRITICAL level
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from sktk.agent.capabilities import Capability
from sktk.agent.contracts import parse_output, serialize_input
from sktk.agent.emitter import AgentEventEmitter
from sktk.agent.filter_runner import FilterRunner
from sktk.agent.filters import AgentFilter
from sktk.agent.hooks import LifecycleHooks
from sktk.agent.providers import LLMProvider
from sktk.agent.runtime import AgentRuntime
from sktk.agent.tools import Tool
from sktk.core.types import TokenUsage
from sktk.observability.events import EventStream
from sktk.session.session import Session

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sktk.agent.router import Router


@runtime_checkable
class Kernel(Protocol):
    """Minimal protocol for kernel-like objects that supply canned responses.

    Satisfied by :class:`sktk.testing.mocks.MockKernel` and any object
    exposing a ``next_response`` method.
    """

    def next_response(self) -> str: ...


@runtime_checkable
class SKAgentLike(Protocol):
    """Minimal protocol for Semantic Kernel agent-like objects.

    Any object with an ``invoke`` method accepting a string prompt.
    """

    async def invoke(self, prompt: str, **kwargs: Any) -> Any: ...


class SKTKAgent:
    """High-level agent wrapping Semantic Kernel's ChatCompletionAgent."""

    __slots__ = (
        "name",
        "instructions",
        "session",
        "capabilities",
        "input_contract",
        "output_contract",
        "filters",
        "tools",
        "hooks",
        "max_iterations",
        "timeout",
        "instructions_version",
        "kernel",
        "service",
        "sk_agent",
        "_runtime",
        "_event_stream",
        "_emitter",
        "_filter_runner",
    )

    def __init__(
        self,
        name: str,
        instructions: str = "",
        *,
        session: Session | None = None,
        capabilities: list[Capability] | None = None,
        input_contract: type[BaseModel] | None = None,
        output_contract: type[BaseModel] | None = None,
        filters: list[AgentFilter] | None = None,
        tools: list[Tool] | None = None,
        hooks: LifecycleHooks | None = None,
        max_iterations: int = 10,
        timeout: float = 60.0,
        instructions_version: str | None = None,
        kernel: Kernel | None = None,
        service: LLMProvider | Router | None = None,
        sk_agent: SKAgentLike | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.session = session
        self.capabilities = capabilities if capabilities is not None else []
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.filters = filters if filters is not None else []
        self.tools = tools if tools is not None else []
        self.hooks = hooks if hooks is not None else LifecycleHooks()
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.instructions_version = instructions_version
        self.kernel = kernel
        self.service = service
        self.sk_agent = sk_agent
        self._runtime = AgentRuntime(self)
        self._event_stream = EventStream()
        self._emitter = AgentEventEmitter(
            agent_name=name,
            instructions=instructions,
            instructions_version=instructions_version,
            event_stream=self._event_stream,
            get_usage=lambda: self._runtime._last_usage,
            get_provider=lambda: self._runtime._last_provider,
        )
        self._filter_runner = FilterRunner(self.filters, name)

    # ------------------------------------------------------------------
    # Metadata properties -- proxy to the runtime so that tests and
    # event emission code can read/write these transparently.
    # ------------------------------------------------------------------

    @property
    def _last_provider(self) -> str | None:
        return self._runtime._last_provider

    @_last_provider.setter
    def _last_provider(self, value: str | None) -> None:
        self._runtime._last_provider = value

    @property
    def _last_usage(self) -> TokenUsage | None:
        return self._runtime._last_usage

    @_last_usage.setter
    def _last_usage(self, value: TokenUsage | None) -> None:
        self._runtime._last_usage = value

    @property
    def _last_response_metadata(self) -> dict[str, Any]:
        return self._runtime._last_response_metadata

    @_last_response_metadata.setter
    def _last_response_metadata(self, value: dict[str, Any]) -> None:
        self._runtime._last_response_metadata = value

    # Public read-only aliases for response metadata.
    @property
    def last_provider(self) -> str | None:
        """The provider name used for the most recent completion."""
        return self._runtime._last_provider

    @property
    def last_usage(self) -> TokenUsage | None:
        """Token usage from the most recent completion."""
        return self._runtime._last_usage

    @property
    def last_response_metadata(self) -> dict[str, Any]:
        """Metadata dict from the most recent completion."""
        return self._runtime._last_response_metadata

    # ------------------------------------------------------------------
    # Factory / builder
    # ------------------------------------------------------------------

    @classmethod
    def with_responses(cls, name: str, responses: list[str], **kwargs: Any) -> SKTKAgent:
        """Create a test agent with scripted responses (no LLM needed)."""
        from sktk.testing.mocks import MockKernel

        mk = MockKernel()
        mk.expect_chat_completion(responses=responses)
        kwargs.setdefault("instructions", "")
        return cls(name=name, kernel=mk, **kwargs)

    @classmethod
    def builder(cls, name: str) -> AgentBuilder:
        """Return a fluent builder for constructing an agent."""
        return AgentBuilder(name)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __rshift__(self, other: Any) -> Any:
        """Support >> operator for pipeline topology DSL."""
        from sktk.team.topology import AgentNode, ParallelNode, SequentialNode

        left = AgentNode(agent=self)
        right: AgentNode | SequentialNode | ParallelNode
        if isinstance(other, list):
            right = ParallelNode(
                [AgentNode(agent=a) if isinstance(a, SKTKAgent) else a for a in other]
            )
        elif isinstance(other, SKTKAgent):
            right = AgentNode(agent=other)
        elif isinstance(other, AgentNode | SequentialNode | ParallelNode):
            right = other
        else:
            return NotImplemented
        return SequentialNode(left, right)

    def __repr__(self) -> str:
        return (
            f"SKTKAgent(name={self.name!r}, instructions={self.instructions!r}, "
            f"max_iterations={self.max_iterations}, timeout={self.timeout})"
        )

    async def __aenter__(self) -> SKTKAgent:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit -- cleanup resources.

        Callers **must** ensure that all in-flight :meth:`invoke` and
        :meth:`invoke_stream` calls have completed (or been cancelled) before
        exiting the ``async with`` block.  This method closes the underlying
        service and session without waiting for concurrent invocations; using
        the agent after ``__aexit__`` has been called results in undefined
        behaviour.

        Note: Tool providers (e.g. MCPToolProvider) are **not** closed here
        because the agent only holds :class:`Tool` dataclass instances, not the
        providers that produced them.  Callers must close tool providers
        separately -- for example via their own ``async with`` block.
        """
        try:
            if self.service and hasattr(self.service, "close"):
                await self.service.close()
        except Exception:
            logger.exception("Error closing service for agent %r", self.name)
        finally:
            try:
                if self.session and hasattr(self.session, "close"):
                    await self.session.close()
            except Exception:
                logger.exception("Error closing session for agent %r", self.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def event_stream(self) -> EventStream:
        """Return the agent event stream for observability consumers."""
        return self._event_stream

    def get_tool(self, name: str) -> Tool | None:
        """Look up a registered tool by name."""
        for t in self.tools:
            if t.name == name:
                return t
        return None

    @staticmethod
    def _validate_tool_args(tool_obj: Tool, kwargs: dict[str, Any]) -> None:
        """Validate kwargs against the tool's declared JSON schema parameters.

        Performs lightweight structural checks without importing jsonschema:
        - All ``required`` parameters must be present in *kwargs*.
        - If ``additionalProperties`` is not explicitly ``True`` and
          ``properties`` is declared, no unexpected keys are allowed.

        Raises :class:`ValueError` when validation fails.
        """
        schema = tool_obj.parameters
        if not schema:
            return

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required parameters are present
        missing = [r for r in required if r not in kwargs]
        if missing:
            raise ValueError(
                f"Tool '{tool_obj.name}' missing required argument(s): {', '.join(missing)}"
            )

        # Check for unexpected parameters (only when properties are declared)
        if properties and not schema.get("additionalProperties", False):
            unexpected = [k for k in kwargs if k not in properties]
            if unexpected:
                raise ValueError(
                    f"Tool '{tool_obj.name}' received unexpected argument(s): {', '.join(unexpected)}"
                )

    async def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Call a registered tool by name."""
        t = self.get_tool(name)
        if t is None:
            raise KeyError(f"Tool '{name}' not registered on agent '{self.name}'")

        self._validate_tool_args(t, kwargs)

        if self._filter_runner.active:
            await self._filter_runner.run_function_call(name, kwargs)

        await self._emitter.emit_tool_call(function=name, arguments=kwargs)
        return await t(**kwargs)

    async def invoke(self, message: str | BaseModel, **kwargs: Any) -> Any:
        """Invoke the agent with a string or typed input."""
        self._runtime.reset_last_response_metadata()
        if isinstance(message, BaseModel):
            prompt = serialize_input(message)
        else:
            prompt = message
        started_at = time.monotonic()
        await self._emitter.emit_thinking()

        await self.hooks.fire_start(self.name, prompt)

        try:
            if self._filter_runner.active:
                prompt = await self._filter_runner.run_input(prompt)

            response_text = await self._get_response(prompt, **kwargs)

            if self._filter_runner.active:
                response_text = await self._filter_runner.run_output(response_text)

            if self.session:
                await self.session.history.append("user", prompt)
                await self.session.history.append("assistant", response_text)

            output: Any = response_text
            if self.output_contract is not None:
                output = parse_output(response_text, self.output_contract)

            await self.hooks.fire_complete(self.name, prompt, output)
            await self._emitter.emit_message(content=response_text)
            await self._emitter.emit_completion(
                result=output,
                duration_seconds=time.monotonic() - started_at,
                total_rounds=self._runtime._last_iterations,
            )
            return output

        except Exception as e:
            await self.hooks.fire_error(self.name, prompt, e)
            raise

    async def invoke_stream(self, message: str | BaseModel, **kwargs: Any) -> AsyncIterator[str]:
        """Invoke the agent and yield response chunks as they arrive.

        Streaming filter semantics
        --------------------------
        * **Per-chunk filters** (``run_output_chunk``) execute on every chunk
          before it is yielded.  They can halt the stream immediately by
          raising :class:`~sktk.core.errors.GuardrailException`.
        * **Post-completion output filters** (``run_output``) run on the fully
          assembled text *after* all chunks have been yielded to the consumer.
          If a post-completion filter returns ``Modify``, the **already-yielded
          chunks are not affected** -- the consumer has already received the
          original text.  The session history, however, stores the
          post-filter (modified) version, so session history and the
          consumer's received text may diverge.
        * For use-cases that require strict output filtering where the
          consumer must only see the post-filter version, use
          :meth:`invoke` instead.

        Hook semantics
        --------------
        ``on_complete`` fires only when both the LLM stream **and** the
        post-completion output filters succeed.  If a post-stream filter
        raises :class:`~sktk.core.errors.GuardrailException`, the
        exception propagates through ``on_error`` instead.  This is
        intentional -- the "complete" event means the full pipeline
        (LLM + filters) succeeded, not just the LLM call.

        Usage::

            async for chunk in agent.invoke_stream("Hello"):
                print(chunk, end="")
        """
        self._runtime.reset_last_response_metadata()
        if isinstance(message, BaseModel):
            prompt = serialize_input(message)
        else:
            prompt = message
        started_at = time.monotonic()
        await self._emitter.emit_thinking()

        await self.hooks.fire_start(self.name, prompt)

        full_response = ""
        output_approved = False
        stream_completed = False
        try:
            if self._filter_runner.active:
                prompt = await self._filter_runner.run_input(prompt)

            # Stream chunks to consumer in real-time. Filters with
            # on_output_chunk get called per-chunk for real-time safety.
            # All filters run post-completion on the full text.
            async for chunk in self._get_response_stream(prompt, **kwargs):
                full_response += chunk
                if self._filter_runner.active:
                    await self._filter_runner.run_output_chunk(chunk, full_response)
                yield chunk

            stream_completed = True

            if self._filter_runner.active:
                full_response = await self._filter_runner.run_output(full_response)

            output_approved = True

            # Post-stream operations run in normal flow, not in finally,
            # because async calls in a finally block of an async generator
            # can silently fail during generator cleanup / GC.
            if self.session:
                await self.session.history.append("user", prompt)
                await self.session.history.append("assistant", full_response)
            await self.hooks.fire_complete(self.name, prompt, full_response)
            await self._emitter.emit_message(content=full_response)
            await self._emitter.emit_completion(
                result=full_response,
                duration_seconds=time.monotonic() - started_at,
                total_rounds=self._runtime._last_iterations,
            )

        except GeneratorExit:
            logger.debug(
                "Agent '%s': stream cancelled by consumer after yielding %d chars",
                self.name,
                len(full_response),
            )
            return
        except Exception as e:
            await self.hooks.fire_error(self.name, prompt, e)
            raise
        finally:
            if stream_completed and not output_approved:
                logger.warning(
                    "Agent '%s': post-completion output filter rejected already-streamed content; "
                    "session history will not contain assistant response for this turn",
                    self.name,
                )

    # ------------------------------------------------------------------
    # LLM dispatch -- delegated to AgentRuntime
    # ------------------------------------------------------------------

    async def _get_response(self, prompt: str, **kwargs: Any) -> str:
        """Get response from LLM, executing tool calls in a loop if needed."""
        return await self._runtime.get_response(prompt, **kwargs)

    async def _get_response_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Get streaming response. Delegates to runtime."""
        async for chunk in self._runtime.get_response_stream(prompt, **kwargs):
            yield chunk

    @staticmethod
    def _coerce_text(result: Any) -> str:
        return AgentRuntime._coerce_text(result)


class AgentBuilder:
    """Fluent builder for constructing SKTKAgent instances."""

    __slots__ = ("_name", "_kwargs")

    def __init__(self, name: str) -> None:
        self._name = name
        self._kwargs: dict[str, Any] = {}

    def instructions(self, instructions: str) -> AgentBuilder:
        self._kwargs["instructions"] = instructions
        return self

    def service(self, service: LLMProvider | Router) -> AgentBuilder:
        self._kwargs["service"] = service
        return self

    def with_safety_filters(self) -> AgentBuilder:
        from sktk.agent.builder import default_safety_filters

        self._kwargs["filters"] = default_safety_filters()
        return self

    def with_tool(self, tool: Tool) -> AgentBuilder:
        self._kwargs.setdefault("tools", [])
        self._kwargs["tools"].append(tool)
        return self

    def with_session(self, session: Session) -> AgentBuilder:
        self._kwargs["session"] = session
        return self

    def with_filter(self, f: AgentFilter) -> AgentBuilder:
        self._kwargs.setdefault("filters", [])
        self._kwargs["filters"].append(f)
        return self

    def timeout(self, timeout: float) -> AgentBuilder:
        self._kwargs["timeout"] = timeout
        return self

    def max_iterations(self, max_iterations: int) -> AgentBuilder:
        self._kwargs["max_iterations"] = max_iterations
        return self

    def output_contract(self, contract: type[BaseModel]) -> AgentBuilder:
        self._kwargs["output_contract"] = contract
        return self

    def input_contract(self, contract: type[BaseModel]) -> AgentBuilder:
        self._kwargs["input_contract"] = contract
        return self

    def build(self) -> SKTKAgent:
        return SKTKAgent(name=self._name, **self._kwargs)
