"""Pattern 1: sequential pipeline.

Each agent receives the previous agent's output as its input.

Usage:
    python examples/concepts/multi_agent/patterns/01_sequential_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from _provider import get_provider

from sktk import SKTKAgent


async def main() -> None:
    provider = get_provider()

    researcher = SKTKAgent(
        name="researcher",
        instructions="You are a research assistant. Find key papers and summarize findings. Be concise.",
        service=provider,
        timeout=30.0,
    )
    analyst = SKTKAgent(
        name="analyst",
        instructions="You are a technical analyst. Analyze the research provided and identify core innovations. Be concise.",
        service=provider,
        timeout=30.0,
    )
    writer = SKTKAgent(
        name="writer",
        instructions="You are a technical writer. Produce a brief markdown summary from the analysis. Be concise.",
        service=provider,
        timeout=30.0,
    )

    pipeline = researcher >> analyst >> writer
    result = await pipeline.run("Survey transformer architectures")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
