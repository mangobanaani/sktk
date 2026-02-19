"""Pattern 5: debate and consensus.

Two agents argue opposing positions and a judge produces a final synthesis.

Usage:
    python examples/concepts/multi_agent/patterns/05_debate_consensus.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from _provider import get_provider

from sktk import BroadcastStrategy, Session, SKTKAgent, SKTKTeam


async def main() -> None:
    provider = get_provider()

    optimist = SKTKAgent(
        name="optimist",
        instructions="You argue the optimistic position on any topic. Be concise but persuasive.",
        service=provider,
        timeout=30.0,
    )
    pessimist = SKTKAgent(
        name="pessimist",
        instructions="You argue the pessimistic position on any topic. Be concise but persuasive.",
        service=provider,
        timeout=30.0,
    )

    debate_team = SKTKTeam(
        agents=[optimist, pessimist],
        strategy=BroadcastStrategy(),
        session=Session(id="debate-pattern"),
    )
    positions = await debate_team.run("Will AI create or destroy jobs?")

    judge = SKTKAgent(
        name="judge",
        instructions="You are an impartial judge. Synthesize opposing positions into a balanced verdict. Be concise.",
        service=provider,
        timeout=30.0,
    )
    verdict = await judge.invoke(f"Synthesize these positions into one verdict: {positions}")

    print("Positions:")
    for idx, position in enumerate(positions, start=1):
        print(f"  {idx}. {position}")
    print("\nConsensus verdict:")
    print(f"  {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
