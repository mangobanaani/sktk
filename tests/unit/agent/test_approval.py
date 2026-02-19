# tests/unit/agent/test_approval.py
import asyncio

import pytest

from sktk.agent.approval import ApprovalGate, AutoApprovalFilter
from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Deny
from sktk.observability.audit import AuditTrail


@pytest.mark.asyncio
async def test_approval_gate_approve():
    gate = ApprovalGate(timeout=5.0)

    async def approve_later():
        await asyncio.sleep(0.01)
        gate.approve()

    asyncio.create_task(approve_later())
    result = await gate.wait_for_approval("agent", "delete", {"id": "123"})
    assert result is True
    assert gate.pending.approved is True


@pytest.mark.asyncio
async def test_approval_gate_deny():
    gate = ApprovalGate(timeout=5.0)

    async def deny_later():
        await asyncio.sleep(0.01)
        gate.deny("too dangerous")

    asyncio.create_task(deny_later())
    result = await gate.wait_for_approval("agent", "delete")
    assert result is False


@pytest.fixture
def audit_trail():
    return AuditTrail()


@pytest.mark.asyncio
async def test_approval_gate_logs_denial(audit_trail):
    gate = ApprovalGate(timeout=1.0, audit_trail=audit_trail)

    task = asyncio.create_task(
        gate.wait_for_approval("agent", "delete_db", {"session_id": "sess-3"})
    )
    await asyncio.sleep(0)
    gate.deny("nope")
    result = await task
    assert result is False
    entries = await audit_trail.query(action="approval_denied")
    assert len(entries) == 1
    assert entries[0].details["pending_action"] == "delete_db"
    assert entries[0].details["reason"] == "nope"


@pytest.mark.asyncio
async def test_approval_gate_timeout():
    gate = ApprovalGate(timeout=0.05)
    result = await gate.wait_for_approval("agent", "delete")
    assert result is False


@pytest.mark.asyncio
async def test_approval_gate_as_filter():
    gate = ApprovalGate(timeout=5.0)

    async def approve_later():
        await asyncio.sleep(0.01)
        gate.approve()

    ctx = FilterContext(
        content="delete_db", stage="function_call", metadata={"function_name": "delete_db"}
    )
    asyncio.create_task(approve_later())
    result = await gate.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_approval_gate_filter_deny():
    gate = ApprovalGate(timeout=5.0)

    async def deny_later():
        await asyncio.sleep(0.01)
        gate.deny("nope")

    ctx = FilterContext(
        content="delete_db", stage="function_call", metadata={"function_name": "delete_db"}
    )
    asyncio.create_task(deny_later())
    result = await gate.on_function_call(ctx)
    assert isinstance(result, Deny)


@pytest.mark.asyncio
async def test_approval_gate_reset():
    gate = ApprovalGate()
    gate.approve()
    await gate.reset()
    assert gate.pending is None


@pytest.mark.asyncio
async def test_auto_approval_safe():
    gate = ApprovalGate()
    auto = AutoApprovalFilter(safe_functions=["search"], gate=gate)
    ctx = FilterContext(
        content="search", stage="function_call", metadata={"function_name": "search"}
    )
    result = await auto.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_auto_approval_input_output():
    gate = ApprovalGate()
    auto = AutoApprovalFilter(safe_functions=["search"], gate=gate)
    ctx = FilterContext(content="hello", stage="input")
    assert isinstance(await auto.on_input(ctx), Allow)
    ctx = FilterContext(content="result", stage="output")
    assert isinstance(await auto.on_output(ctx), Allow)


@pytest.mark.asyncio
async def test_approval_gate_on_input_and_output_always_allow():
    gate = ApprovalGate()
    assert isinstance(await gate.on_input(FilterContext(content="x", stage="input")), Allow)
    assert isinstance(await gate.on_output(FilterContext(content="y", stage="output")), Allow)


@pytest.mark.asyncio
async def test_consecutive_wait_for_approval_does_not_return_stale_result():
    """Calling wait_for_approval() twice on the same gate should require fresh approval each time."""
    gate = ApprovalGate(timeout=5.0)

    # First approval cycle
    async def approve_first():
        await asyncio.sleep(0.01)
        gate.approve()

    asyncio.create_task(approve_first())
    first = await gate.wait_for_approval("agent", "action1")
    assert first is True

    # Second approval cycle -- must NOT return immediately with the old result
    received_second = asyncio.Event()

    async def deny_second():
        # Wait until the gate is actually waiting (pending is set)
        for _ in range(50):
            if gate.pending is not None and gate.pending.action == "action2":
                break
            await asyncio.sleep(0.01)
        gate.deny("denied second time")
        received_second.set()

    asyncio.create_task(deny_second())
    second = await gate.wait_for_approval("agent", "action2")
    assert second is False
    assert gate.pending.approved is False


@pytest.mark.asyncio
async def test_auto_approval_unsafe_function_delegates_to_gate():
    gate = ApprovalGate()
    auto = AutoApprovalFilter(safe_functions=["search"], gate=gate)
    calls = {}

    async def fake_gate(context):
        calls["context"] = context
        return Deny(reason="denied")

    gate.on_function_call = fake_gate  # type: ignore[method-assign]
    ctx = FilterContext(
        content="delete_db",
        stage="function_call",
        metadata={"function_name": "delete_db"},
    )

    result = await auto.on_function_call(ctx)
    assert isinstance(result, Deny)
    assert calls["context"] is ctx
