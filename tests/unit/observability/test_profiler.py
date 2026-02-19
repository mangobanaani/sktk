# tests/unit/observability/test_profiler.py
import time

import pytest

from sktk.observability.profiler import AgentProfiler, SessionRecorder


@pytest.mark.asyncio
async def test_profiler_measure():
    profiler = AgentProfiler()
    async with profiler.measure("test_op"):
        time.sleep(0.01)
    assert len(profiler.entries) == 1
    assert profiler.entries[0].label == "test_op"
    assert profiler.entries[0].duration_ms > 0


@pytest.mark.asyncio
async def test_profiler_record():
    profiler = AgentProfiler()
    await profiler.record("op1", 100.0)
    await profiler.record("op2", 200.0)
    assert profiler.total_ms() == 300.0


@pytest.mark.asyncio
async def test_profiler_summary():
    profiler = AgentProfiler()
    await profiler.record("llm_call", 100.0)
    await profiler.record("llm_call", 150.0)
    await profiler.record("tool_call", 50.0)
    summary = profiler.summary()
    assert summary["total_ms"] == 300.0
    assert summary["entries"] == 3
    assert summary["breakdown"]["llm_call"]["count"] == 2
    assert summary["breakdown"]["tool_call"]["total_ms"] == 50.0


@pytest.mark.asyncio
async def test_profiler_clear():
    profiler = AgentProfiler()
    await profiler.record("op", 100.0)
    await profiler.clear()
    assert len(profiler.entries) == 0


@pytest.mark.asyncio
async def test_session_recorder():
    recorder = SessionRecorder()
    await recorder.record_turn("user", "Hello", agent_name="support")
    await recorder.record_turn("assistant", "Hi there!", agent_name="support")
    assert recorder.turn_count == 2
    entries = recorder.replay()
    assert len(entries) == 2
    assert entries[0].role == "user"
    assert entries[1].content == "Hi there!"


@pytest.mark.asyncio
async def test_session_replay_from():
    recorder = SessionRecorder()
    await recorder.record_turn("user", "msg1")
    await recorder.record_turn("assistant", "msg2")
    await recorder.record_turn("user", "msg3")
    entries = recorder.replay_from(2)
    assert len(entries) == 2
    assert entries[0].content == "msg2"


@pytest.mark.asyncio
async def test_session_to_dict():
    recorder = SessionRecorder()
    await recorder.record_turn("user", "Hello", agent_name="bot")
    d = recorder.to_dict()
    assert len(d) == 1
    assert d[0]["role"] == "user"
    assert d[0]["agent_name"] == "bot"


@pytest.mark.asyncio
async def test_session_clear():
    recorder = SessionRecorder()
    await recorder.record_turn("user", "Hello")
    await recorder.clear()
    assert recorder.turn_count == 0
    assert len(recorder.replay()) == 0


@pytest.mark.asyncio
async def test_profiler_trims_oldest_when_exceeding_max_entries():
    profiler = AgentProfiler(max_entries=3)
    for i in range(5):
        await profiler.record(f"op{i}", float(i) * 10)
    entries = profiler.entries
    assert len(entries) == 3
    assert entries[0].label == "op2"
    assert entries[-1].label == "op4"


@pytest.mark.asyncio
async def test_session_recorder_trims_oldest_when_exceeding_max_entries():
    recorder = SessionRecorder(max_entries=3)
    for i in range(5):
        await recorder.record_turn("user", f"msg{i}")
    entries = recorder.replay()
    assert len(entries) == 3
    assert entries[0].content == "msg2"
    assert entries[-1].content == "msg4"
