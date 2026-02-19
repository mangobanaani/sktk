"""Persistent session example.

Demonstrates using Session with SQLite-backed history
to persist conversations across runs.

Usage:
    python examples/getting_started/02_persistent_session.py
"""

import asyncio
from pathlib import Path

from sktk import Session, SQLiteHistory


async def main() -> None:
    db_path = str(Path(__file__).parent / "demo_sessions.db")

    history = SQLiteHistory(db_path=db_path, session_id="demo-session")
    await history.initialize()

    session = Session(id="demo-session", history=history)

    await session.history.append("user", "What is Python?")
    await session.history.append(
        "assistant",
        "Python is a high-level programming language.",
    )

    messages = await session.history.get()
    print(f"Session has {len(messages)} messages:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content']}")

    forked = await session.history.fork("forked-session")
    print(f"\nForked session has {len(await forked.get())} messages")

    await history.close()
    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
