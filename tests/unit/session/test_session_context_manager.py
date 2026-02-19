# tests/unit/session/test_session_context_manager.py
import pytest

from sktk.session.session import Session


@pytest.mark.asyncio
async def test_session_context_manager():
    async with Session(id="s1") as session:
        assert session.id == "s1"
        await session.history.append("user", "hello")
    msgs = await session.history.get()
    assert len(msgs) == 1


@pytest.mark.asyncio
async def test_session_close():
    session = Session(id="s1")
    await session.close()  # Should not raise
