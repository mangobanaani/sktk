"""Approval, permissions, and rate limits.

Demonstrates three governance controls in one agent:
1) PermissionPolicy for allow/deny tool access.
2) AutoApprovalFilter + ApprovalGate for human-in-the-loop actions.
3) RateLimitPolicy to throttle invoke() calls.

Usage:
    python examples/concepts/agent/approval_permissions_rate_limits.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _provider import get_provider

from sktk import GuardrailException, SKTKAgent, tool
from sktk.agent.approval import ApprovalGate, AutoApprovalFilter
from sktk.agent.permissions import PermissionPolicy, RateLimitPolicy


@tool(description="Read docs for a named topic")
async def read_docs(topic: str) -> str:
    return f"Docs summary for {topic}: checklist, caveats, and rollout tips."


@tool(description="Deploy a release to an environment")
async def deploy_release(environment: str) -> str:
    return f"Deployment scheduled to {environment}."


@tool(description="Dangerous operation used for policy-deny demo")
async def drop_database() -> str:
    return "Database dropped"


async def approve_once(gate: ApprovalGate) -> None:
    """Simulate a human approver that approves one pending request."""
    while gate.pending is None:
        await asyncio.sleep(0.01)
    pending = gate.pending
    print(f"Human approval requested: action={pending.action}, details={pending.details}")
    gate.approve()


async def main() -> None:
    provider = get_provider()
    gate = ApprovalGate(timeout=3.0)

    filters = [
        PermissionPolicy(allow=["read_docs", "deploy_release"]),
        AutoApprovalFilter(safe_functions=["read_docs"], gate=gate),
        RateLimitPolicy(max_calls=2, window_seconds=60.0),
    ]

    agent = SKTKAgent(
        name="governed-agent",
        instructions="Follow governance controls.",
        service=provider,
        timeout=30.0,
        filters=filters,
        tools=[read_docs, deploy_release, drop_database],
    )

    print("=== Permission allowlist + auto-approved tool ===")
    docs = await agent.call_tool("read_docs", topic="on-call runbook")
    print(f"read_docs -> {docs}")

    print("\n=== Human approval for sensitive tool ===")
    approver_task = asyncio.create_task(approve_once(gate))
    deploy = await agent.call_tool("deploy_release", environment="staging")
    await approver_task
    print(f"deploy_release -> {deploy}")

    print("\n=== Permission deny example ===")
    try:
        await agent.call_tool("drop_database")
    except GuardrailException as exc:
        print(f"drop_database blocked: {exc.reason}")

    print("\n=== Rate limit on invoke() ===")
    print(await agent.invoke("request #1"))
    print(await agent.invoke("request #2"))
    try:
        await agent.invoke("request #3")
    except GuardrailException as exc:
        print(f"request #3 blocked: {exc.reason}")


if __name__ == "__main__":
    asyncio.run(main())
