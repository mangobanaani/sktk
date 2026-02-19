# tests/unit/agent/test_providers.py
from __future__ import annotations

from typing import Any

import pytest

from sktk.agent.providers import (
    AnthropicClaudeProvider,
    AzureOpenAIProvider,
    CompletionResult,
    GeminiProvider,
    LLMProvider,
    LocalLLMProvider,
    ProviderRegistry,
    _coerce_usage,
    _usage_from_claude,
    _usage_from_gemini,
    _usage_from_openai,
    create_provider,
    extract_tool_calls,
    get_registry,
    normalize_completion_result,
    register_provider,
)
from sktk.core.types import TokenUsage, maybe_await


class FakeOpenAIProvider:
    """Fake OpenAI provider for testing."""

    def __init__(self, api_key: str = "", model: str = "gpt-4") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"openai: {messages[-1]['content']}"


class FakeAnthropicProvider:
    """Fake Anthropic provider for testing."""

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"anthropic: {messages[-1]['content']}"


def test_provider_protocol():
    provider = FakeOpenAIProvider()
    assert isinstance(provider, LLMProvider)


def test_registry_register_and_create():
    registry = ProviderRegistry()
    registry.register("openai", FakeOpenAIProvider)
    provider = registry.create("openai", api_key="test-key")
    assert provider.name == "openai"


def test_registry_unknown_provider():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="Unknown provider"):
        registry.create("nonexistent")


def test_registry_available():
    registry = ProviderRegistry()
    registry.register("openai", FakeOpenAIProvider)
    registry.register("anthropic", FakeAnthropicProvider)
    assert set(registry.available) == {"openai", "anthropic"}


def test_registry_multiple_providers():
    registry = ProviderRegistry()
    registry.register("openai", FakeOpenAIProvider)
    registry.register("anthropic", FakeAnthropicProvider)
    p1 = registry.create("openai")
    p2 = registry.create("anthropic")
    assert p1.name == "openai"
    assert p2.name == "anthropic"


@pytest.mark.asyncio
async def test_provider_complete():
    provider = FakeOpenAIProvider(api_key="test")
    result = await provider.complete([{"role": "user", "content": "hello"}])
    assert "openai" in result


def test_global_registry():
    registry = get_registry()
    assert isinstance(registry, ProviderRegistry)


def test_global_register_and_create():
    register_provider("test-provider", FakeOpenAIProvider)
    provider = create_provider("test-provider", api_key="key")
    assert provider.name == "openai"


class _FakeAzureChoice:
    def __init__(self, content: str) -> None:
        self.message = type("msg", (), {"content": content})


class _FakeAzureResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeAzureChoice(content)]
        self.usage = type("usage", (), {"prompt_tokens": 4, "completion_tokens": 6})()


class _FakeAzureChat:
    def completions(self): ...

    def completions_create(self, messages, model): ...

    def __getattr__(self, item):
        # allow chat.completions.create shape
        if item == "completions":
            return type(
                "Completions",
                (),
                {"create": lambda self, **kwargs: _FakeAzureResponse("azure hello")},
            )()
        raise AttributeError


class _FakeAzureClient:
    def __init__(self, content: str = "azure hi") -> None:
        self.chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "Comp", (), {"create": lambda self, **kwargs: _FakeAzureResponse(content)}
                )()
            },
        )()


class _FakeClaudeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeClaudeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeClaudeContent(text)]
        self.usage = type("usage", (), {"input_tokens": 3, "output_tokens": 5})()


class _FakeClaudeClient:
    def __init__(self, text: str = "claude hi") -> None:
        self.messages = type(
            "Msgs", (), {"create": lambda self, **kwargs: _FakeClaudeResponse(text)}
        )()


@pytest.mark.asyncio
async def test_azure_openai_provider_complete():
    client = _FakeAzureClient("azure result")
    provider = AzureOpenAIProvider(client=client, deployment="gpt-4o")
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, CompletionResult)
    assert "azure" in result.text
    assert result.usage is not None
    assert result.usage.total_tokens == 10
    assert provider.name == "azure-openai"


@pytest.mark.asyncio
async def test_claude_provider_complete():
    client = _FakeClaudeClient("claude result")
    provider = AnthropicClaudeProvider(client=client, model="claude-3-sonnet")
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, CompletionResult)
    assert "claude result" in result.text
    assert result.usage is not None
    assert result.usage.total_tokens == 8
    assert provider.name == "claude"


def test_registry_registers_new_providers():
    registry = ProviderRegistry()
    registry.register("azure-openai", AzureOpenAIProvider)
    registry.register("claude", AnthropicClaudeProvider)
    assert set(registry.available) == {"azure-openai", "claude"}


class _FakeGeminiClient:
    def __init__(self, text: str = "gemini hi") -> None:
        self.responses = text

    class GenerativeModel:
        def __init__(self, outer, text: str) -> None:
            self._text = text
            self._outer = outer

        async def generate_content(self, messages, **kwargs):
            class _Resp:
                def __init__(self, t: str) -> None:
                    self.text = t
                    self.usage_metadata = type(
                        "usage",
                        (),
                        {
                            "prompt_token_count": 2,
                            "candidates_token_count": 4,
                        },
                    )()

            return _Resp(self._text)

    def generative_model(self, model: str):
        return self.GenerativeModel(self, self.responses)


class _FakeLocalModel:
    def __init__(self, reply: str = "local hi") -> None:
        self.reply = reply

    async def chat(self, messages, **kwargs):
        return self.reply


@pytest.mark.asyncio
async def test_gemini_provider_complete():
    client = _FakeGeminiClient("gemini result")
    provider = GeminiProvider(client=client, model="gpt-gemini")
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, CompletionResult)
    assert "gemini result" in result.text
    assert result.usage is not None
    assert result.usage.total_tokens == 6
    assert provider.name == "gemini"


@pytest.mark.asyncio
async def test_local_provider_complete():
    client = _FakeLocalModel("local result")
    provider = LocalLLMProvider(client=client)
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert isinstance(result, CompletionResult)
    assert "local result" in result.text
    assert result.usage is None
    assert provider.name == "local"


def test_normalize_completion_result_tuple_with_usage():
    normalized = normalize_completion_result(
        (
            "ok",
            {
                "provider": "demo",
                "usage": {
                    "prompt_tokens": "2",
                    "completion_tokens": "3",
                    "total_cost_usd": "0.25",
                },
            },
        )
    )
    assert normalized.text == "ok"
    assert normalized.metadata == {"provider": "demo"}
    assert normalized.usage is not None
    assert normalized.usage.total_tokens == 5
    assert normalized.usage.total_cost_usd == pytest.approx(0.25)


def test_coerce_usage_invalid_values_return_none():
    assert _coerce_usage({"prompt_tokens": "bad", "completion_tokens": "2"}) is None
    assert _coerce_usage({"prompt_tokens": 1}) is None


def test_usage_helpers_return_none_when_usage_missing():
    assert _usage_from_openai(object()) is None
    assert _usage_from_claude(type("R", (), {"usage": None})()) is None
    assert _usage_from_gemini(type("R", (), {"usage_metadata": None})()) is None


def test_provider_initializers_require_client():
    with pytest.raises(ValueError, match="requires a client instance"):
        AzureOpenAIProvider(client=None, deployment="d1")
    with pytest.raises(ValueError, match="requires a client instance"):
        AnthropicClaudeProvider(client=None)
    with pytest.raises(ValueError, match="requires a client instance"):
        GeminiProvider(client=None)
    with pytest.raises(ValueError, match="requires a client instance"):
        LocalLLMProvider(client=None)


@pytest.mark.asyncio
async def test_gemini_provider_missing_model_factory_raises():
    class NoModelClient:
        pass

    provider = GeminiProvider(client=NoModelClient())
    with pytest.raises(ValueError, match="missing generative_model"):
        await provider.complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def testmaybe_await_returns_non_awaitable_as_is():
    marker = object()
    assert await maybe_await(marker) is marker


@pytest.mark.asyncio
async def test_llm_provider_protocol_stub_methods_are_executable():
    assert LLMProvider.name.fget(object()) is None
    assert await LLMProvider.complete(object(), []) is None


def test_coerce_usage_returns_token_usage_instance_unchanged():
    usage = TokenUsage(prompt_tokens=1, completion_tokens=2, total_cost_usd=0.5)
    assert _coerce_usage(usage) is usage


# ---------------------------------------------------------------
# extract_tool_calls tests
# ---------------------------------------------------------------


def test_extract_tool_calls_openai_format():
    """OpenAI-style response with choices[0].message.tool_calls."""
    fn_obj = type("Fn", (), {"name": "search", "arguments": '{"query": "hello"}'})()
    tc_obj = type("TC", (), {"id": "call_1", "function": fn_obj})()
    msg_obj = type("Msg", (), {"tool_calls": [tc_obj]})()
    choice = type("Choice", (), {"message": msg_obj})()
    response = type("Response", (), {"choices": [choice]})()

    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "hello"}


def test_extract_tool_calls_anthropic_format():
    """Anthropic-style response with content blocks of type 'tool_use'."""
    block = type(
        "Block",
        (),
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "calculator",
            "input": {"expression": "2+2"},
        },
    )()
    response = type("Response", (), {"content": [block]})()

    # Must not have choices attribute to fall into Anthropic path
    assert not hasattr(response, "choices") or getattr(response, "choices", None) is None

    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].id == "toolu_1"
    assert calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "2+2"}


def test_extract_tool_calls_no_tool_calls_returns_empty():
    """Response with no tool calls returns empty list."""
    # OpenAI-style with no tool_calls on message
    msg_obj = type("Msg", (), {"tool_calls": None})()
    choice = type("Choice", (), {"message": msg_obj})()
    response = type("Response", (), {"choices": [choice]})()

    calls = extract_tool_calls(response)
    assert calls == []


def test_extract_tool_calls_malformed_response_returns_empty():
    """Completely unrecognized response shape returns empty list."""
    response = type("Response", (), {})()
    calls = extract_tool_calls(response)
    assert calls == []


def test_extract_tool_calls_openai_malformed_arguments():
    """OpenAI-style with unparseable JSON arguments defaults to empty dict."""
    fn_obj = type("Fn", (), {"name": "search", "arguments": "not valid json"})()
    tc_obj = type("TC", (), {"id": "call_2", "function": fn_obj})()
    msg_obj = type("Msg", (), {"tool_calls": [tc_obj]})()
    choice = type("Choice", (), {"message": msg_obj})()
    response = type("Response", (), {"choices": [choice]})()

    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {}
