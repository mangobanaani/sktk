# tests/unit/session/test_sqlite_history.py
import pytest

from sktk.session.backends.sqlite import SQLiteHistory


@pytest.mark.asyncio
async def test_sqlite_history_append_and_get(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "hello")
    await h.append("assistant", "hi there")
    messages = await h.get()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_get_with_limit(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    for i in range(5):
        await h.append("user", f"msg{i}")
    messages = await h.get(limit=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "msg3"
    assert messages[1]["content"] == "msg4"
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_get_with_limit_larger_than_total(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "msg0")
    await h.append("assistant", "msg1")
    messages = await h.get(limit=5)
    assert len(messages) == 2
    assert messages[0]["content"] == "msg0"
    assert messages[1]["content"] == "msg1"
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_get_with_roles(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "q")
    await h.append("assistant", "a")
    messages = await h.get(roles=["user"])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_clear(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "hello")
    await h.clear()
    assert len(h) == 0
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_fork(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "hello")
    forked = await h.fork("s2")
    messages = await forked.get()
    assert len(messages) == 1
    await forked.append("user", "more")
    assert len(forked) == 2
    assert len(h) == 1
    await forked.close()
    await h.close()


@pytest.mark.asyncio
async def test_sqlite_history_persistence(tmp_path):
    db_path = str(tmp_path / "test.db")
    h1 = SQLiteHistory(db_path=db_path, session_id="s1")
    await h1.initialize()
    await h1.append("user", "persisted")
    await h1.close()
    h2 = SQLiteHistory(db_path=db_path, session_id="s1")
    await h2.initialize()
    messages = await h2.get()
    assert len(messages) == 1
    assert messages[0]["content"] == "persisted"
    await h2.close()


@pytest.mark.asyncio
async def test_sqlite_history_close(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "hello")
    await h.close()
    with pytest.raises(RuntimeError, match="not initialized"):
        await h.append("user", "fail")


@pytest.mark.asyncio
async def test_sqlite_history_auto_initializes():
    h = SQLiteHistory(db_path=":memory:", session_id="s1")
    # Auto-initializes on first use without explicit initialize() call
    await h.append("user", "hello")
    msgs = await h.get()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_sqlite_history_initialize_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    h = SQLiteHistory(db_path=db_path, session_id="s1")
    await h.initialize()
    await h.append("user", "hello")
    await h.initialize()  # second call is no-op
    assert len(h) == 1
    await h.close()
