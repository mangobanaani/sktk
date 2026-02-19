"""Multi-agent team example.

Demonstrates creating a team of agents with round-robin
coordination and event streaming.

Usage:
    python examples/concepts/multi_agent/team_with_round_robin.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _provider import get_provider

from sktk import RoundRobinStrategy, Session, SKTKAgent, SKTKTeam


async def main() -> None:
    provider = get_provider()
    session = Session(id="team-demo")

    roles = {
        "researcher": "You are a financial researcher. Find key data points about earnings. Be concise.",
        "analyst": "You are a financial analyst. Analyze the data provided and identify key drivers. Be concise.",
        "writer": "You are a report writer. Produce a brief markdown summary of the analysis. Be concise.",
    }
    agents = [
        SKTKAgent(
            name=name,
            instructions=instructions,
            service=provider,
            timeout=30.0,
            session=session,
        )
        for name, instructions in roles.items()
    ]

    team = SKTKTeam(
        agents=agents,
        strategy=RoundRobinStrategy(),
        session=session,
        max_rounds=3,
    )

    print("Team executing task: 'Analyze Q3 earnings'\n")
    async for event in team.stream("Analyze Q3 earnings"):
        if event.kind == "message":
            print(f"[{event.kind}] {event.agent}: {event.content[:80]}")
        elif event.kind == "completion":
            print(f"[{event.kind}] Duration: {event.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
