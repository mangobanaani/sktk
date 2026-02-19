import pytest

from sktk.session.backends.memory import InMemoryBlackboard, InMemoryHistory
from sktk.session.session import Session


class ClosableHistory:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def test_session_creation_defaults():
    s = Session(id="sess-1")
    assert s.id == "sess-1"
    assert s.tenant_id is None
    assert isinstance(s.history, InMemoryHistory)
    assert isinstance(s.blackboard, InMemoryBlackboard)


def test_session_creation_custom_backends():
    h = InMemoryHistory()
    bb = InMemoryBlackboard()
    s = Session(id="sess-2", history=h, blackboard=bb, tenant_id="t1")
    assert s.history is h
    assert s.tenant_id == "t1"


@pytest.mark.asyncio
async def test_session_history_operations():
    s = Session(id="sess-3")
    await s.history.append("user", "hello")
    messages = await s.history.get()
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_session_blackboard_operations():
    from pydantic import BaseModel

    class Note(BaseModel):
        text: str

    s = Session(id="sess-4")
    await s.blackboard.set("note", Note(text="important"))
    got = await s.blackboard.get("note", Note)
    assert got is not None
    assert got.text == "important"


@pytest.mark.asyncio
async def test_session_context_manager_closes_history_when_available():
    history = ClosableHistory()
    async with Session(id="sess-close-ctx", history=history):
        pass
    assert history.close_calls == 1


@pytest.mark.asyncio
async def test_session_close_calls_history_close_when_available():
    history = ClosableHistory()
    s = Session(id="sess-close", history=history)
    await s.close()
    assert history.close_calls == 1


class ClosableBlackboard:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class FailingHistory:
    """History whose close() always raises."""

    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        raise RuntimeError("history close failed")


@pytest.mark.asyncio
async def test_session_close_calls_both_history_and_blackboard_close():
    history = ClosableHistory()
    blackboard = ClosableBlackboard()
    s = Session(id="sess-both", history=history, blackboard=blackboard)
    await s.close()
    assert history.close_calls == 1
    assert blackboard.close_calls == 1


@pytest.mark.asyncio
async def test_session_close_calls_blackboard_even_when_history_raises():
    history = FailingHistory()
    blackboard = ClosableBlackboard()
    s = Session(id="sess-exc-safety", history=history, blackboard=blackboard)
    # history.close() raises, but blackboard.close() must still be called
    with pytest.raises(RuntimeError, match="history close failed"):
        await s.close()
    assert history.close_calls == 1
    assert blackboard.close_calls == 1
