"""Provider registry + router fallback.

Demonstrates how to:
1) Register provider factories.
2) Build providers from configuration.
3) Route calls with fallback when the primary provider fails.

Usage:
    python examples/concepts/agent/provider_router_fallback.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import anthropic
from _provider import MODEL

from sktk.agent.builder import build_safe_agent
from sktk.agent.providers import AnthropicClaudeProvider, ProviderRegistry
from sktk.agent.router import FallbackPolicy, Router
from sktk.core.secrets import FileSecretsProvider


async def main() -> None:
    root = Path(__file__).resolve().parents[3]  # project root
    secrets = FileSecretsProvider(root / ".env")
    key = secrets.require("ANTHROPIC_API_KEY")

    registry = ProviderRegistry()
    registry.register("claude", AnthropicClaudeProvider)
    print(f"Available provider factories: {registry.available}")

    # Primary uses a nonexistent model (will fail), backup uses the real model.
    client = anthropic.AsyncAnthropic(api_key=key)
    primary = registry.create(
        "claude", client=client, model="nonexistent-model-999", max_tokens=256
    )
    backup = registry.create("claude", client=client, model=MODEL, max_tokens=256)

    router = Router(providers=[primary, backup], policy=FallbackPolicy())

    # build_safe_agent wires router + default safety filters.
    agent = build_safe_agent(
        name="router-demo",
        instructions="Summarize user requests concisely.",
        router=router,
    )

    question = "Summarize why fallback policies are useful."
    result = await agent.invoke(question)

    print(f"Question : {question}")
    print(f"Answer   : {result}")
    print(f"Provider : {getattr(agent, '_last_provider', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
