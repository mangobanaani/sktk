"""Blackboard: shared state between agents.

The blackboard is a typed key-value store that lives on a Session.
Agents write intermediate results, and other agents read them --
enabling coordination without direct message passing.

Usage:
    python examples/concepts/session/blackboard_shared_state.py
"""

import asyncio

from pydantic import BaseModel

from sktk import Session


# Typed models stored on the blackboard
class ResearchFindings(BaseModel):
    topic: str
    key_facts: list[str]


class AnalysisResult(BaseModel):
    summary: str
    confidence: float


async def main() -> None:
    session = Session(id="blackboard-demo")
    bb = session.blackboard

    # 1) Agent A writes findings to the blackboard
    print("=== Agent A: Researcher writes findings ===")
    findings = ResearchFindings(
        topic="quantum computing",
        key_facts=[
            "Qubits can be in superposition",
            "Entanglement enables parallel computation",
            "Error correction is the main engineering challenge",
        ],
    )
    await bb.set("research:quantum", findings)
    print(f"  Wrote: {findings}")

    # 2) Agent B reads the findings and writes analysis
    print("\n=== Agent B: Analyst reads and analyzes ===")
    read_back = await bb.get("research:quantum", ResearchFindings)
    assert read_back is not None
    print(f"  Read {len(read_back.key_facts)} facts about '{read_back.topic}'")

    analysis = AnalysisResult(
        summary=f"Found {len(read_back.key_facts)} key insights on {read_back.topic}.",
        confidence=0.87,
    )
    await bb.set("analysis:quantum", analysis)

    # 3) List all keys, fetch everything under a prefix
    print("\n=== Blackboard state ===")
    all_keys = await bb.keys()
    print(f"  Keys: {all_keys}")

    research_data = await bb.get_all("research:")
    print(f"  research:* -> {research_data}")

    # 4) Delete a key
    deleted = await bb.delete("research:quantum")
    print(f"\n  Deleted 'research:quantum': {deleted}")
    print(f"  Remaining keys: {await bb.keys()}")


if __name__ == "__main__":
    asyncio.run(main())
