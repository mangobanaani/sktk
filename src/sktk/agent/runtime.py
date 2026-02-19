"""AgentRuntime -- internal LLM dispatch engine extracted from SKTKAgent.

This module contains the low-level LLM interaction logic: message
construction, service invocation, tool-call loops, streaming, and
response metadata tracking.  SKTKAgent delegates to AgentRuntime for
all LLM communication while retaining ownership of the public API,
filter pipelines, hooks, session management, and event emission.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Awaitable

from sktk.agent.contracts import output_json_schema
from sktk.agent.providers import (
    CompletionResult,
    extract_tool_calls,
    normalize_completion_result,
)
from sktk.core.types import TokenUsage, maybe_await
from sktk.observability.metrics import record_metric

if TYPE_CHECKING:
    from sktk.agent.agent import SKTKAgent

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Internal runtime that handles LLM dispatch, tool-call loops, and streaming.

    This is not part of the public API.  SKTKAgent creates and owns an
    ``AgentRuntime`` instance and delegates the actual LLM communication
    to it.  The runtime holds a back-reference to the owning agent so it
    can read mutable fields (kernel, service, tools, etc.) that may be
    set after construction.
    """

    __slots__ = (
        "_agent",
        "_last_provider",
        "_last_usage",
        "_last_response_metadata",
        "_last_iterations",
    )

    def __init__(self, agent: SKTKAgent) -> None:
        self._agent = agent
        self._last_provider: str | None = None
        self._last_usage: TokenUsage | None = None
        self._last_response_metadata: dict[str, Any] = {}
        self._last_iterations: int = 1

    # ------------------------------------------------------------------
    # Response metadata
    # ------------------------------------------------------------------

    def reset_last_response_metadata(self) -> None:
        self._last_provider = None
        self._last_usage = None
        self._last_response_metadata = {}
        self._last_iterations = 1

    def record_response_metadata(self, metadata: dict[str, Any]) -> None:
        usage = metadata.get("usage")
        if isinstance(usage, TokenUsage):
            self._last_usage = usage
        elif isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                self._last_usage = TokenUsage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_cost_usd=usage.get("total_cost_usd"),
                )
        self._last_provider = metadata.get("provider")
        self._last_response_metadata = metadata

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def resolve_timeout(self, kwargs: dict[str, Any]) -> float | None:
        if "timeout" in kwargs:
            value = kwargs.pop("timeout")
        else:
            value = self._agent.timeout
        if value is None:
            return None
        timeout = float(value)
        if timeout <= 0:
            raise TimeoutError(f"Agent '{self._agent.name}' timeout must be > 0, got {timeout}")
        return timeout

    async def await_with_timeout(self, call: Awaitable[Any], timeout: float | None) -> Any:
        if timeout is None:
            return await call
        try:
            return await asyncio.wait_for(call, timeout=timeout)
        except TimeoutError as exc:  # pragma: no cover - branch depends on scheduler timing
            raise TimeoutError(
                f"Agent '{self._agent.name}' timed out after {timeout:.2f}s"
            ) from exc

    async def iterate_stream_with_timeout(
        self,
        stream: AsyncIterator[Any],
        timeout: float | None,
    ) -> AsyncIterator[Any]:
        if timeout is None:
            async for chunk in stream:
                yield chunk
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Agent '{self._agent.name}' timed out after {timeout:.2f}s")
            try:
                chunk = await asyncio.wait_for(anext(stream), timeout=remaining)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Agent '{self._agent.name}' timed out after {timeout:.2f}s"
                ) from exc
            yield chunk

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    async def build_messages(self, prompt: str) -> list[dict[str, str]]:
        agent = self._agent
        messages: list[dict[str, str]] = []
        if agent.instructions:
            messages.append({"role": "system", "content": agent.instructions})
        if agent.session:
            history = await agent.session.history.get()
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Core LLM dispatch
    # ------------------------------------------------------------------

    async def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Get response from LLM, executing tool calls in a loop if needed."""
        agent = self._agent

        if agent.kernel is not None and hasattr(agent.kernel, "next_response"):
            return agent.kernel.next_response()

        timeout = self.resolve_timeout(kwargs)

        if agent.service is not None:
            messages = await self.build_messages(prompt)
            return await self._tool_call_loop(messages, timeout, **kwargs)

        if agent.sk_agent is not None and hasattr(agent.sk_agent, "invoke"):
            result = await self.await_with_timeout(
                maybe_await(agent.sk_agent.invoke(prompt, **kwargs)),
                timeout=timeout,
            )
            return self._coerce_text(result)

        raise NotImplementedError(
            "Real SK integration requires kernel= or service= parameter. "
            "For testing, use SKTKAgent.with_responses()."
        )

    async def _tool_call_loop(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM-tool loop until a text response or max_iterations."""
        agent = self._agent

        if agent.max_iterations <= 0:
            return ""

        self._last_iterations = 1
        tools_schema = [t.to_schema() for t in agent.tools] if agent.tools else None
        completion = CompletionResult(text="")

        loop = asyncio.get_running_loop()
        deadline = (loop.time() + timeout) if timeout is not None else None

        def _remaining() -> float | None:
            if deadline is None:
                return None
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Agent '{agent.name}' timed out after {timeout:.2f}s")
            return remaining

        for _iteration in range(agent.max_iterations):
            logger.debug(
                "Tool call loop iteration %d/%d",
                _iteration + 1,
                agent.max_iterations,
            )
            self._last_iterations = _iteration + 1
            completion = await self._call_service(messages, _remaining(), tools_schema, **kwargs)

            if not completion.tool_calls:
                return completion.text

            # Append assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text or "",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in completion.tool_calls
                    ],
                }
            )

            # Execute tool calls concurrently with timeout protection
            tool_tasks = [self._execute_tool_call(tc, agent) for tc in completion.tool_calls]
            remaining = _remaining()
            if remaining is not None:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tool_tasks),
                        timeout=remaining,
                    )
                except TimeoutError as exc:
                    logger.warning(
                        "Agent '%s': tool execution timed out after %.2fs",
                        agent.name,
                        timeout,
                    )
                    raise TimeoutError(
                        f"Agent '{agent.name}' tool execution timed out after {timeout:.2f}s"
                    ) from exc
            else:
                results = await asyncio.gather(*tool_tasks)
            messages.extend(results)

        # max_iterations exceeded -- return last text or empty
        logger.warning(
            "Agent '%s' exceeded max_iterations (%d); last response contained %d pending tool call(s)",
            agent.name,
            agent.max_iterations,
            len(completion.tool_calls) if completion.tool_calls else 0,
        )
        return completion.text if completion.text else ""

    async def _execute_tool_call(
        self,
        tc: Any,
        agent: SKTKAgent,
    ) -> dict[str, str]:
        """Execute a single tool call and return a tool-role message dict."""
        try:
            result = await agent.call_tool(tc.name, **tc.arguments)
            return {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
        except KeyError:
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Error: tool '{tc.name}' not found",
            }
        except (TypeError, ValueError) as e:
            # Argument validation errors — safe to expose so the LLM can retry
            logger.debug("Tool '%s' validation error: %s", tc.name, e)
            return {"role": "tool", "tool_call_id": tc.id, "content": f"Error: {e}"}
        except TimeoutError as e:
            logger.warning("Tool '%s' timeout: %s", tc.name, e)
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Error: tool execution timed out",
            }
        except (ConnectionError, OSError) as e:
            logger.error("Tool '%s' connection failed: %s", tc.name, e, exc_info=True)
            return {"role": "tool", "tool_call_id": tc.id, "content": "Error: network failure"}
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tc.name, e, exc_info=True)
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Error: tool execution failed",
            }

    async def _call_service(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None,
        tools_schema: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Single LLM call through the service, returning a CompletionResult."""
        agent = self._agent
        extra: dict[str, Any] = {}
        if tools_schema:
            extra["tools"] = tools_schema

        start_time = time.time()
        try:
            # Constrained decoding: pass JSON schema for structured output
            if agent.output_contract is not None:
                schema = output_json_schema(agent.output_contract)
                extra["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema,
                }

            # Merge kwargs, but let extra (framework-set) keys take precedence
            merged = {**kwargs, **extra}

            if "response_format" in merged:
                logger.debug(
                    "Passing response_format to service; provider must support structured output"
                )

            if hasattr(agent.service, "complete_with_metadata"):
                call = agent.service.complete_with_metadata(messages, **merged)
                text, metadata = await self.await_with_timeout(call, timeout=timeout)
                meta = dict(metadata)
                self.record_response_metadata(meta)

                # Extract tool calls: the text may already be a CompletionResult,
                # or metadata may carry a "tool_calls" list.
                tool_calls: list[Any] = []
                if isinstance(text, CompletionResult):
                    tool_calls = text.tool_calls
                    text_str = text.text
                else:
                    text_str = str(text)

                if not tool_calls and "tool_calls" in meta:
                    raw_tc = meta["tool_calls"]
                    if isinstance(raw_tc, list):
                        from sktk.agent.providers import ToolCallRequest

                        for tc in raw_tc:
                            if isinstance(tc, ToolCallRequest):
                                tool_calls.append(tc)
                            elif isinstance(tc, dict):
                                tool_calls.append(
                                    ToolCallRequest(
                                        id=tc.get("id", ""),
                                        name=tc.get("name", ""),
                                        arguments=tc.get("arguments", {}),
                                    )
                                )

                if not tool_calls and hasattr(text, "choices"):
                    tool_calls = extract_tool_calls(text)

                duration = time.time() - start_time
                record_metric(
                    "agent.service_call.duration",
                    duration,
                    {"agent": agent.name, "provider": "unknown"},
                )
                logger.debug("Service call completed in %.3fs", duration)

                return CompletionResult(text=text_str, tool_calls=tool_calls)

            call = agent.service.complete(messages, **merged)
            raw = await self.await_with_timeout(call, timeout=timeout)
            completion = normalize_completion_result(raw)

            # Try to extract tool calls from the raw response
            tool_calls = completion.tool_calls
            if not tool_calls and hasattr(raw, "choices"):
                tool_calls = extract_tool_calls(raw)

            metadata = dict(completion.metadata)
            if completion.usage is not None:
                metadata["usage"] = completion.usage
            if hasattr(agent.service, "name"):
                metadata.setdefault("provider", getattr(agent.service, "name", None))
            self.record_response_metadata(metadata)

            duration = time.time() - start_time
            record_metric(
                "agent.service_call.duration",
                duration,
                {"agent": agent.name, "provider": metadata.get("provider", "unknown")},
            )
            logger.debug("Service call completed in %.3fs", duration)

            return CompletionResult(
                text=completion.text,
                usage=completion.usage,
                metadata=completion.metadata,
                tool_calls=tool_calls,
            )
        except Exception:
            duration = time.time() - start_time
            record_metric(
                "agent.service_call.duration", duration, {"agent": agent.name, "status": "error"}
            )
            logger.debug("Service call failed after %.3fs", duration)
            raise

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _build_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build extra kwargs for streaming calls (tools, response_format).

        Mirrors the logic in ``_call_service`` so that streaming providers
        receive the same tool schemas and structured-output constraints as
        non-streaming calls.  Tool-call loops are not supported during
        streaming; a warning is logged when tools are present.
        """
        agent = self._agent
        extra: dict[str, Any] = {}
        if agent.tools:
            tools_schema = [t.to_schema() for t in agent.tools]
            extra["tools"] = tools_schema
            logger.debug(
                "Agent '%s': tools passed to streaming provider but tool-call "
                "loops are not supported in streaming mode",
                agent.name,
            )
        if agent.output_contract is not None:
            schema = output_json_schema(agent.output_contract)
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": schema,
            }
        return {**kwargs, **extra}

    async def get_response_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Get streaming response from LLM."""
        agent = self._agent

        if agent.kernel is not None and hasattr(agent.kernel, "next_response"):
            response = agent.kernel.next_response()
            words = response.split(" ")
            for i, word in enumerate(words):
                yield word if i == len(words) - 1 else word + " "
            return

        timeout = self.resolve_timeout(kwargs)

        if agent.service is not None and hasattr(agent.service, "stream_with_metadata"):
            messages = await self.build_messages(prompt)
            merged = self._build_stream_kwargs(kwargs)
            stream_obj = await self.await_with_timeout(
                maybe_await(agent.service.stream_with_metadata(messages, **merged)),
                timeout=timeout,
            )
            stream, metadata = self.unpack_stream_result(stream_obj)
            self.record_response_metadata(metadata)
            async for chunk in self.iterate_stream_with_timeout(stream, timeout=timeout):
                text, chunk_meta = self.coerce_stream_chunk(chunk)
                if chunk_meta:
                    self.record_response_metadata({**self._last_response_metadata, **chunk_meta})
                yield text
            return

        if agent.service is not None and hasattr(agent.service, "stream"):
            messages = await self.build_messages(prompt)
            merged = self._build_stream_kwargs(kwargs)
            stream = agent.service.stream(messages, **merged)
            async for chunk in self.iterate_stream_with_timeout(stream, timeout=timeout):
                text, chunk_meta = self.coerce_stream_chunk(chunk)
                if chunk_meta:
                    self.record_response_metadata({**self._last_response_metadata, **chunk_meta})
                yield text
            return

        # Default stream fallback: perform a single completion and yield coarse chunks.
        response = await self.get_response(prompt, timeout=timeout, **kwargs)
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
        return

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------

    def unpack_stream_result(
        self,
        stream_obj: Any,
    ) -> tuple[AsyncIterator[Any], dict[str, Any]]:
        if isinstance(stream_obj, tuple) and len(stream_obj) == 2:
            stream, metadata = stream_obj
            if hasattr(stream, "__aiter__"):
                return stream, dict(metadata)
        if hasattr(stream_obj, "__aiter__"):
            return stream_obj, {}
        raise TypeError(
            "stream_with_metadata must return an async iterator or (iterator, metadata)"
        )

    def coerce_stream_chunk(self, chunk: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            text, metadata = chunk
            return str(text), dict(metadata)
        completion = normalize_completion_result(chunk)
        metadata = dict(completion.metadata)
        if completion.usage is not None:
            metadata["usage"] = completion.usage
        return completion.text, metadata

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            return str(result.content)
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)
