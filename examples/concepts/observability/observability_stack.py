"""Observability: token tracking, audit trail, profiling.

Shows the three pillars of SKTK observability working together
so you can monitor cost, compliance, and performance in production.

Usage:
    python examples/concepts/observability/observability_stack.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _provider import get_provider

from sktk import SKTKAgent, TokenUsage
from sktk.observability import AgentProfiler, AuditTrail, TokenTracker
from sktk.observability.metrics import PricingModel


async def main() -> None:
    provider = get_provider()

    agent = SKTKAgent(
        name="analyst",
        instructions="You are a financial analyst. Analyze earnings data. Be concise.",
        service=provider,
        timeout=30.0,
    )

    # -- 1) Token tracking with cost attribution --
    print("=== Token Tracking ===")
    pricing = PricingModel(
        prices={
            "claude-haiku": {"prompt": 0.00025, "completion": 0.00125},
            "claude-sonnet": {"prompt": 0.003, "completion": 0.015},
        }
    )
    tracker = TokenTracker(pricing=pricing)

    result = await agent.invoke("Analyze Q3 earnings")

    # Record real usage from the provider if available, else use representative values
    usage = agent._last_usage or TokenUsage(prompt_tokens=150, completion_tokens=80)
    tracker.record(
        agent_name="analyst",
        session_id="s1",
        model="claude-haiku",
        usage=usage,
    )

    await agent.invoke("What were the key drivers?")
    usage2 = agent._last_usage or TokenUsage(prompt_tokens=200, completion_tokens=100)
    tracker.record(
        agent_name="analyst",
        session_id="s1",
        model="claude-haiku",
        usage=usage2,
    )

    total_usage = tracker.get_usage(session_id="s1")
    print(f"  Total tokens  : {total_usage.total_tokens}")
    print(f"  Prompt tokens : {total_usage.prompt_tokens}")
    print(f"  Completion    : {total_usage.completion_tokens}")
    print(f"  Estimated cost: ${total_usage.total_cost_usd:.4f}")

    # -- 2) Audit trail with tamper-evident hashing --
    print("\n=== Audit Trail ===")
    audit = AuditTrail()

    await audit.record(
        action="invoke",
        agent_name="analyst",
        session_id="s1",
        user_id="u1",
        details={"input": "Analyze Q3 earnings", "output_len": len(str(result))},
    )
    await audit.record(
        action="invoke",
        agent_name="analyst",
        session_id="s1",
        user_id="u1",
        details={"input": "What were the key drivers?"},
    )

    entries = await audit.query(session_id="s1")
    for e in entries:
        print(f"  [{e.action}] {e.agent_name} hash={e.entry_hash} prev={e.previous_hash}")

    # Verify the hash chain hasn't been tampered with
    is_valid = audit.verify_chain(entries)
    print(f"  Chain integrity: {'VALID' if is_valid else 'TAMPERED'}")

    # -- 3) Performance profiling --
    print("\n=== Profiling ===")
    profiler = AgentProfiler()

    agent2 = SKTKAgent(
        name="profiled",
        instructions="You are a helpful assistant. Be concise.",
        service=provider,
        timeout=30.0,
    )
    async with profiler.measure("invoke"):
        await agent2.invoke("test")

    async with profiler.measure("post_process"):
        await asyncio.sleep(0.001)  # simulate post-processing

    summary = profiler.summary()
    print(f"  Total time   : {summary['total_ms']:.1f}ms")
    print(f"  Entries      : {summary['entries']}")
    for label, stats in summary["breakdown"].items():
        print(f"  {label:14s}: {stats['total_ms']:.1f}ms ({stats['count']} calls)")


if __name__ == "__main__":
    asyncio.run(main())
