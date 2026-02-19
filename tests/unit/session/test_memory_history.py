import pytest

from sktk.session.backends.memory import InMemoryHistory


@pytest.mark.asyncio
async def test_append_and_get():
    h = InMemoryHistory()
    await h.append("user", "hello")
    await h.append("assistant", "hi there")
    messages = await h.get()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_with_limit():
    h = InMemoryHistory()
    await h.append("user", "msg1")
    await h.append("assistant", "msg2")
    await h.append("user", "msg3")
    messages = await h.get(limit=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "msg2"
    assert messages[1]["content"] == "msg3"


@pytest.mark.asyncio
async def test_get_with_role_filter():
    h = InMemoryHistory()
    await h.append("user", "q1")
    await h.append("assistant", "a1")
    await h.append("user", "q2")
    messages = await h.get(roles=["user"])
    assert len(messages) == 2
    assert all(m["role"] == "user" for m in messages)


@pytest.mark.asyncio
async def test_clear():
    h = InMemoryHistory()
    await h.append("user", "hello")
    await h.clear()
    assert len(h) == 0


@pytest.mark.asyncio
async def test_fork():
    h = InMemoryHistory()
    await h.append("user", "hello")
    await h.append("assistant", "hi")
    forked = await h.fork("new-session")
    assert len(forked) == 2
    await forked.append("user", "followup")
    assert len(forked) == 3
    assert len(h) == 2


@pytest.mark.asyncio
async def test_len():
    h = InMemoryHistory()
    assert len(h) == 0
    await h.append("user", "hello")
    assert len(h) == 1


@pytest.mark.asyncio
async def test_metadata_preserved():
    h = InMemoryHistory()
    await h.append("user", "hello", metadata={"source": "api"})
    messages = await h.get()
    assert messages[0]["metadata"] == {"source": "api"}
