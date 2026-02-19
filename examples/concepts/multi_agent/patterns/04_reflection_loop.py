"""Pattern 4: reflection / self-critique loop.

A generator drafts output, a critic gives feedback, then the generator refines.

Usage:
    python examples/concepts/multi_agent/patterns/04_reflection_loop.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from _provider import get_provider

from sktk import SKTKAgent


async def main() -> None:
    provider = get_provider()

    generator = SKTKAgent(
        name="generator",
        instructions="You are a code generator. Write clean Python code for the given task. Be concise.",
        service=provider,
        timeout=30.0,
    )
    critic = SKTKAgent(
        name="critic",
        instructions="You are a code critic. Review code and list specific improvements needed. Be concise.",
        service=provider,
        timeout=30.0,
    )

    draft = await generator.invoke("Write a sort function")
    feedback = await critic.invoke(f"Review this code: {draft}")
    refined = await generator.invoke(f"Improve based on feedback: {feedback}")
    print("Draft:")
    print(draft)
    print("\nFeedback:")
    print(feedback)
    print("\nRefined:")
    print(refined)


if __name__ == "__main__":
    asyncio.run(main())
