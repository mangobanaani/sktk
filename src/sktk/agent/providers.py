"""LLM provider abstraction -- Factory pattern for swappable backends.

Defines a Protocol for LLM providers and a registry-based factory
so OpenAI can be swapped for Anthropic, Azure, local models, etc.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sktk.core.types import TokenUsage, maybe_await

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult:
    """Normalized provider completion payload.

    Contains completion text and optional usage/metadata for observability.
    When the LLM requests tool calls, ``tool_calls`` is non-empty and
    ``text`` may be empty.
    """

    text: str
    usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider backends.

    Implement this to integrate any LLM service (OpenAI, Anthropic,
    Azure, local models, etc.) with SKTK agents.
    """

    @property
    def name(self) -> str:
        """Provider identifier (e.g. 'openai', 'anthropic')."""
        ...

    async def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> str | CompletionResult | tuple[str, dict[str, Any]]:
        """Send messages and return a completion payload.

        Implementations should return :class:`CompletionResult` for best
        results (usage tracking, metadata, tool calls).  Returning a plain
        ``str`` is accepted for simplicity; it will be wrapped automatically
        via :func:`normalize_completion_result`.
        """
        ...


class ProviderRegistry:
    """Registry + factory for LLM providers.

    Uses ``threading.Lock`` because ``register`` and ``create`` are called
    from synchronous setup code.  Not intended for use inside async hot paths.

    Usage:
        registry = ProviderRegistry()
        registry.register("openai", OpenAIProvider)
        provider = registry.create("openai", api_key="sk-...")
    """

    def __init__(self) -> None:
        self._factories: dict[str, type[LLMProvider]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: type[LLMProvider]) -> None:
        """Register a provider class under a name."""
        with self._lock:
            self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> LLMProvider:
        """Create a provider instance by registered name."""
        with self._lock:
            factory = self._factories.get(name)
            available = list(self._factories.keys())
        if factory is None:
            raise KeyError(f"Unknown provider '{name}'. Available: {available}")
        return factory(**kwargs)

    @property
    def available(self) -> list[str]:
        """List registered provider names."""
        with self._lock:
            return list(self._factories.keys())


# Global default registry
_default_registry = ProviderRegistry()


def register_provider(name: str, factory: type[LLMProvider]) -> None:
    """Register a provider in the default global registry."""
    _default_registry.register(name, factory)


def create_provider(name: str, **kwargs: Any) -> LLMProvider:
    """Create a provider from the default global registry."""
    return _default_registry.create(name, **kwargs)


def get_registry() -> ProviderRegistry:
    """Get the default provider registry."""
    return _default_registry


class AzureOpenAIProvider:
    """Azure OpenAI chat provider.

    Expects an injected Azure OpenAI client with a `chat.completions.create` method
    matching the official SDK. Keeps networking out of the library by requiring the
    caller to supply the client instance.
    """

    def __init__(
        self, client: Any, deployment: str, model: str | None = None, **kwargs: Any
    ) -> None:
        if client is None:
            raise ValueError("AzureOpenAIProvider requires a client instance")
        self._client = client
        self._deployment = deployment
        self._model = model or deployment
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "azure-openai"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        params = {**self._kwargs, **kwargs}
        response = await maybe_await(
            self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **params,
            )
        )
        text = response.choices[0].message.content
        usage = _usage_from_openai(response)
        return CompletionResult(
            text=text,
            usage=usage,
            metadata={"model": self._model, "deployment": self._deployment},
        )

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await maybe_await(self._client.close())


class AnthropicClaudeProvider:
    """Anthropic Claude chat provider."""

    def __init__(self, client: Any, model: str = "claude-3-sonnet-20240229", **kwargs: Any) -> None:
        if client is None:
            raise ValueError("AnthropicClaudeProvider requires a client instance")
        self._client = client
        self._model = model
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "claude"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        """Send messages to Claude and return a CompletionResult.

        When ``response_format`` contains a ``json_schema``, structured output
        is implemented via Claude's tool_use mechanism: a dedicated tool is
        created from the schema and ``tool_choice`` is locked to that tool.

        .. warning::

            If the caller also passes ``tools`` (e.g. agent tools), they are
            appended to the tools list but **will never be invoked** because
            ``tool_choice`` forces the structured output tool.  A warning is
            logged when this situation is detected.
        """
        params = {**self._kwargs, **kwargs}
        response_format = params.pop("response_format", None)
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        if system_parts and "system" not in params:
            params["system"] = "\n".join(system_parts)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": non_system,
            "max_tokens": params.pop("max_tokens", 4096),
        }

        if response_format and "json_schema" in response_format:
            schema = response_format["json_schema"]
            # Claude uses tool_use for structured output
            call_kwargs["tools"] = [
                {
                    "name": schema.get("name", "structured_output"),
                    "description": "Respond with structured output",
                    "input_schema": schema.get("schema", {}),
                }
            ]
            call_kwargs["tool_choice"] = {
                "type": "tool",
                "name": schema.get("name", "structured_output"),
            }

        # Merge structured output tool with caller's tools if both present
        if "tools" in call_kwargs and "tools" in params:
            logger.warning(
                "Agent tools are present alongside output_contract; "
                "tool_choice forces structured output tool, agent tools will not be invoked"
            )
            call_kwargs["tools"] = call_kwargs["tools"] + params.pop("tools")
        # Pass remaining params through
        call_kwargs.update(params)

        response = await maybe_await(self._client.messages.create(**call_kwargs))

        # Extract text and tool calls from all content blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                # Structured output tool_use blocks: if this is the forced
                # structured-output tool, treat its input as text content.
                if response_format and "json_schema" in response_format:
                    text_parts.append(json.dumps(getattr(block, "input", {})))
                else:
                    tool_calls.append(
                        ToolCallRequest(
                            id=getattr(block, "id", ""),
                            name=getattr(block, "name", ""),
                            arguments=getattr(block, "input", {}),
                        )
                    )
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        text = "\n".join(text_parts) if text_parts else ""
        usage = _usage_from_claude(response)
        return CompletionResult(
            text=text,
            usage=usage,
            metadata={"model": self._model},
            tool_calls=tool_calls,
        )

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await maybe_await(self._client.close())


class GeminiProvider:
    """Google Gemini chat provider (client injected)."""

    def __init__(self, client: Any, model: str = "gemini-pro", **kwargs: Any) -> None:
        if client is None:
            raise ValueError("GeminiProvider requires a client instance")
        self._client = client
        self._model = model
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "gemini"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        params = {**self._kwargs, **kwargs}
        response_format = params.pop("response_format", None)

        model = getattr(self._client, "generative_model", None) or getattr(
            self._client, "GenerativeModel", None
        )
        if model is None:
            raise ValueError("Gemini client missing generative_model")
        gm = model(self._model) if callable(model) else model

        # Apply structured output via Gemini's generation_config
        if response_format and "json_schema" in response_format:
            schema = response_format["json_schema"]
            generation_config = params.pop("generation_config", {})
            if not isinstance(generation_config, dict):
                generation_config = {}
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = schema.get("schema", {})
            params["generation_config"] = generation_config

        response = await maybe_await(gm.generate_content(messages, **params))
        text = response.text
        usage = _usage_from_gemini(response)
        return CompletionResult(text=text, usage=usage, metadata={"model": self._model})

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await maybe_await(self._client.close())


class LocalLLMProvider:
    """Local LLM provider; client must expose a chat(messages) coroutine."""

    def __init__(self, client: Any, name: str = "local", **kwargs: Any) -> None:
        if client is None:
            raise ValueError("LocalLLMProvider requires a client instance")
        self._client = client
        self._name = name
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        params = {**self._kwargs, **kwargs}
        response = await maybe_await(self._client.chat(messages, **params))
        return CompletionResult(text=str(response))

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await maybe_await(self._client.close())


def normalize_completion_result(
    result: str | CompletionResult | tuple[str, dict[str, Any]],
) -> CompletionResult:
    """Normalize provider responses to CompletionResult."""
    if isinstance(result, CompletionResult):
        return result
    if isinstance(result, tuple):
        text, metadata = result
        meta = dict(metadata)
        usage = _coerce_usage(meta.pop("usage", None))
        return CompletionResult(text=text, usage=usage, metadata=meta)
    return CompletionResult(text=str(result))


def extract_tool_calls(response: Any) -> list[ToolCallRequest]:
    """Extract tool call requests from a raw provider response.

    Supports OpenAI-style ``choices[0].message.tool_calls`` and
    Anthropic-style ``content`` blocks with ``type == "tool_use"``.
    """
    calls: list[ToolCallRequest] = []

    # OpenAI / Azure style
    choices = getattr(response, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        tc_list = getattr(msg, "tool_calls", None) or []
        for tc in tc_list:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw_args = getattr(fn, "arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(
                ToolCallRequest(
                    id=getattr(tc, "id", ""),
                    name=getattr(fn, "name", ""),
                    arguments=args,
                )
            )
        return calls

    # Anthropic style
    content_blocks = getattr(response, "content", None)
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                calls.append(
                    ToolCallRequest(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=getattr(block, "input", {}),
                    )
                )

    return calls


def _coerce_usage(value: Any) -> TokenUsage | None:
    """Best-effort conversion from provider usage payloads to TokenUsage."""
    if value is None:
        return None
    if isinstance(value, TokenUsage):
        return value

    prompt = getattr(value, "prompt_tokens", None)
    completion = getattr(value, "completion_tokens", None)
    cost = getattr(value, "total_cost_usd", None)
    if prompt is None and isinstance(value, dict):
        prompt = value.get("prompt_tokens")
        completion = value.get("completion_tokens")
        cost = value.get("total_cost_usd")

    if prompt is None or completion is None:
        return None

    try:
        prompt_i = int(prompt)
        completion_i = int(completion)
    except (TypeError, ValueError):
        return None

    cost_f = float(cost) if cost is not None else None
    return TokenUsage(
        prompt_tokens=prompt_i,
        completion_tokens=completion_i,
        total_cost_usd=cost_f,
    )


def _usage_from_openai(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    return _coerce_usage(usage)


def _usage_from_claude(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    payload = {
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "completion_tokens": getattr(usage, "output_tokens", None),
    }
    return _coerce_usage(payload)


def _usage_from_gemini(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    payload = {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
    }
    return _coerce_usage(payload)


class OpenAIProvider:
    """OpenAI chat provider (non-Azure).

    Expects an injected ``openai.AsyncOpenAI`` (or synchronous ``OpenAI``)
    client.  Keeps networking out of the library by requiring the caller to
    supply the client instance.

    Usage:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="sk-...")
        provider = OpenAIProvider(client=client, model="gpt-4o")
    """

    def __init__(self, client: Any, model: str = "gpt-4o", **kwargs: Any) -> None:
        if client is None:
            raise ValueError("OpenAIProvider requires a client instance")
        self._client = client
        self._model = model
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        params = {**self._kwargs, **kwargs}
        response = await maybe_await(
            self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **params,
            )
        )
        text = response.choices[0].message.content
        usage = _usage_from_openai(response)
        return CompletionResult(
            text=text,
            usage=usage,
            metadata={"model": self._model},
        )

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await maybe_await(self._client.close())


# Register built-in providers with the default registry
register_provider("openai", OpenAIProvider)
register_provider("azure-openai", AzureOpenAIProvider)
register_provider("claude", AnthropicClaudeProvider)
register_provider("gemini", GeminiProvider)
register_provider("local", LocalLLMProvider)
