"""Integration tests for AnthropicClaudeProvider against the real Claude API."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_provider_returns_nonempty_text(claude_provider):
    result = await claude_provider.complete([{"role": "user", "content": "Say ok"}])
    assert result.text.strip(), "Expected non-empty text from provider"


async def test_provider_returns_token_usage(claude_provider):
    result = await claude_provider.complete([{"role": "user", "content": "Say ok"}])
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


async def test_provider_metadata_contains_model(claude_provider):
    result = await claude_provider.complete([{"role": "user", "content": "Say ok"}])
    assert "model" in result.metadata
    assert "claude" in result.metadata["model"]


async def test_provider_extracts_system_message(claude_provider):
    result = await claude_provider.complete(
        [
            {"role": "system", "content": "Always reply with exactly: SYSTEM_OK"},
            {"role": "user", "content": "Hello"},
        ]
    )
    normalized = result.text.upper().replace(" ", "")
    assert "SYSTEM_OK" in normalized or "SYSTEMOK" in normalized


async def test_provider_handles_multi_turn(claude_provider):
    result = await claude_provider.complete(
        [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]
    )
    assert "alice" in result.text.lower()


async def test_provider_name_is_claude(claude_provider):
    assert claude_provider.name == "claude"
