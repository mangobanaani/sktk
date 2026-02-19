"""Pattern 2: parallel fan-out / fan-in.

A planner fans a task out to multiple workers, then a synthesizer merges results.

Usage:
    python examples/concepts/multi_agent/patterns/02_parallel_fanout_fanin.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from _provider import get_provider

from sktk import SKTKAgent
from sktk.team.topology import AgentNode


async def main() -> None:
    provider = get_provider()

    planner = SKTKAgent(
        name="planner",
        instructions="You are a research planner. Identify the key search terms for the given topic. Be concise.",
        service=provider,
        timeout=30.0,
    )
    searcher_a = SKTKAgent(
        name="arxiv-searcher",
        instructions="You are an academic paper searcher. Summarize relevant papers you find. Be concise.",
        service=provider,
        timeout=30.0,
    )
    searcher_b = SKTKAgent(
        name="web-searcher",
        instructions="You are a web content searcher. Summarize relevant blog posts and articles. Be concise.",
        service=provider,
        timeout=30.0,
    )
    synthesizer = SKTKAgent(
        name="synthesizer",
        instructions="You are a research synthesizer. Combine the provided sources into a unified summary. Be concise.",
        service=provider,
        timeout=30.0,
    )

    pipeline = AgentNode(agent=planner) >> [searcher_a, searcher_b] >> synthesizer
    result = await pipeline.run("Research RAG techniques")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
