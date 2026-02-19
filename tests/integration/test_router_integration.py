"""Integration tests for Router with real Claude API."""

from __future__ import annotations

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.providers import AnthropicClaudeProvider
from sktk.agent.router import FallbackPolicy, Router

from .conftest import MAX_TOKENS

pytestmark = pytest.mark.integration


async def test_router_fallback_completes(claude_router):
    result, meta = await claude_router.complete_with_metadata(
        [{"role": "user", "content": "Say ok"}]
    )
    assert len(result.strip()) > 0
    assert meta.get("provider") == "claude"


async def test_router_complete_shortcut(claude_router):
    result = await claude_router.complete([{"role": "user", "content": "Say ok"}])
    assert isinstance(result, str)
    assert len(result.strip()) > 0


async def test_router_skips_broken_provider(anthropic_client, claude_provider):
    broken = AnthropicClaudeProvider(
        client=anthropic_client,
        model="nonexistent-model-xyz",
        max_tokens=MAX_TOKENS,
    )
    router = Router(
        providers=[broken, claude_provider],
        policy=FallbackPolicy(),
    )
    result = await router.complete([{"role": "user", "content": "Say ok"}])
    assert isinstance(result, str)
    assert len(result.strip()) > 0


async def test_router_with_agent(claude_router):
    agent = SKTKAgent(
        name="router-agent",
        instructions="Be concise.",
        service=claude_router,
        timeout=30.0,
    )
    result = await agent.invoke("What is 2+2? Just the number.")
    assert "4" in str(result)
