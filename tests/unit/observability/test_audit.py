# tests/unit/observability/test_audit.py
import pytest

from sktk.observability.audit import AuditEntry, AuditTrail


@pytest.fixture
def audit():
    return AuditTrail()


@pytest.mark.asyncio
async def test_record_entry(audit):
    entry = await audit.record(
        action="invoke",
        agent_name="analyst",
        session_id="s1",
        user_id="u1",
        correlation_id="c1",
        details={"input": "test query"},
    )
    assert entry.action == "invoke"
    assert entry.agent_name == "analyst"
    assert entry.entry_hash != ""


@pytest.mark.asyncio
async def test_query_by_session(audit):
    await audit.record(action="invoke", agent_name="a", session_id="s1")
    await audit.record(action="invoke", agent_name="b", session_id="s2")
    entries = await audit.query(session_id="s1")
    assert len(entries) == 1
    assert entries[0].agent_name == "a"


@pytest.mark.asyncio
async def test_query_by_agent(audit):
    await audit.record(action="invoke", agent_name="analyst", session_id="s1")
    await audit.record(action="invoke", agent_name="writer", session_id="s1")
    entries = await audit.query(agent_name="analyst")
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_query_by_action(audit):
    await audit.record(action="invoke", agent_name="a", session_id="s1")
    await audit.record(action="filter_deny", agent_name="a", session_id="s1")
    entries = await audit.query(action="filter_deny")
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_hash_chain_integrity(audit):
    await audit.record(action="step1", agent_name="a", session_id="s1")
    await audit.record(action="step2", agent_name="a", session_id="s1")
    await audit.record(action="step3", agent_name="a", session_id="s1")
    entries = await audit.query()
    assert audit.verify_chain(entries)


@pytest.mark.asyncio
async def test_hash_chain_tamper_detection(audit):
    await audit.record(action="step1", agent_name="a", session_id="s1")
    await audit.record(action="step2", agent_name="a", session_id="s1")
    entries = await audit.query()

    # Tamper with an entry by creating a modified copy
    tampered = AuditEntry(
        timestamp=entries[0].timestamp,
        action="TAMPERED",
        agent_name=entries[0].agent_name,
        session_id=entries[0].session_id,
        user_id=entries[0].user_id,
        correlation_id=entries[0].correlation_id,
        details=entries[0].details,
        outcome=entries[0].outcome,
        duration_ms=entries[0].duration_ms,
        previous_hash=entries[0].previous_hash,
        entry_hash=entries[0].entry_hash,
    )
    assert not audit.verify_chain([tampered, entries[1]])


@pytest.mark.asyncio
async def test_audit_entry_to_dict(audit):
    entry = await audit.record(action="test", agent_name="a", session_id="s1")
    d = entry.to_dict()
    assert d["action"] == "test"
    assert "entry_hash" in d


@pytest.mark.asyncio
async def test_audit_with_outcome_and_duration(audit):
    entry = await audit.record(
        action="invoke",
        agent_name="a",
        session_id="s1",
        outcome="error",
        duration_ms=150.5,
    )
    assert entry.outcome == "error"
    assert entry.duration_ms == 150.5


@pytest.mark.asyncio
async def test_query_with_limit(audit):
    for i in range(10):
        await audit.record(action=f"step{i}", agent_name="a", session_id="s1")
    entries = await audit.query(limit=3)
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_hash_chain_detects_previous_hash_mismatch(audit):
    await audit.record(action="step1", agent_name="a", session_id="s1")
    await audit.record(action="step2", agent_name="a", session_id="s1")
    entries = await audit.query()
    second = entries[1]
    mismatched = AuditEntry(
        timestamp=second.timestamp,
        action=second.action,
        agent_name=second.agent_name,
        session_id=second.session_id,
        user_id=second.user_id,
        correlation_id=second.correlation_id,
        details=second.details,
        outcome=second.outcome,
        duration_ms=second.duration_ms,
        previous_hash="wrong-prev-hash",
        entry_hash=second.entry_hash,
    )
    assert not audit.verify_chain([entries[0], mismatched])


@pytest.mark.asyncio
async def test_audit_trail_trims_oldest_when_exceeding_max_entries():
    trail = AuditTrail(max_entries=3)
    for i in range(5):
        await trail.record(action=f"step{i}", agent_name="a", session_id="s1")
    entries = await trail.query()
    assert len(entries) == 3
    assert entries[0].action == "step2"
    assert entries[-1].action == "step4"
