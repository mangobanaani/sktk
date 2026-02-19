"""Pattern 3: supervisor / worker hierarchy.

A supervisor strategy delegates sequentially to workers and aggregates outcomes.

Usage:
    python examples/concepts/multi_agent/patterns/03_supervisor_worker.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from _provider import get_provider

from sktk import RoundRobinStrategy, Session, SKTKAgent, SKTKTeam


async def main() -> None:
    provider = get_provider()

    coder = SKTKAgent(
        name="coder",
        instructions="You are a software engineer. Write clean, concise code for the given task.",
        service=provider,
        timeout=30.0,
    )
    tester = SKTKAgent(
        name="tester",
        instructions="You are a QA engineer. Review the code provided and report test results. Be concise.",
        service=provider,
        timeout=30.0,
    )
    reviewer = SKTKAgent(
        name="reviewer",
        instructions="You are a code reviewer. Evaluate code quality and provide approval or feedback. Be concise.",
        service=provider,
        timeout=30.0,
    )

    team = SKTKTeam(
        agents=[coder, tester, reviewer],
        strategy=RoundRobinStrategy(),
        session=Session(id="supervisor"),
        max_rounds=3,
    )
    result = await team.run("Implement fibonacci with tests")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
