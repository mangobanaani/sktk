"""Basic SKTK agent example.

Demonstrates creating and invoking a simple SKTKAgent with
minimal configuration using the Claude API.

Usage:
    python examples/getting_started/01_basic_agent.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _provider import get_provider

from sktk import SKTKAgent, context_scope


async def main() -> None:
    provider = get_provider()
    agent = SKTKAgent(
        name="assistant",
        instructions="You are a helpful assistant. Be concise.",
        service=provider,
        timeout=30.0,
    )

    async with context_scope(tenant_id="demo", user_id="user-1"):
        print(f"Agent '{agent.name}' created successfully")
        question = "What is Python?"
        answer = await agent.invoke(question)
        print(f"Q: {question}")
        print(f"A: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
