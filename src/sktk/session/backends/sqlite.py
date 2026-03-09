"""SQLite-backed persistent session backends."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

from sktk.session.history import ConversationHistory


class SQLiteHistory(ConversationHistory):
    """SQLite-backed persistent conversation history."""

    def __init__(self, db_path: str, session_id: str) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._conn: sqlite3.Connection | None = None
        self._count = 0
        self._closed = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self) -> sqlite3.Connection:
        """Return the active connection, auto-initializing on first use.

        Must be called while holding ``self._lock``.
        """
        if self._closed:
            raise RuntimeError("SQLiteHistory not initialized; call initialize() first")
        if self._conn is None:
            await self._initialize_locked()
        return self._conn  # type: ignore[return-value]

    async def initialize(self) -> None:
        """Create the messages table if needed and count existing rows."""
        async with self._lock:
            if self._conn is not None:
                return
            await self._initialize_locked()

    async def _initialize_locked(self) -> None:
        session_id = self._session_id

        def _init() -> tuple[sqlite3.Connection, int]:
            # check_same_thread=False is safe here because the asyncio.Lock
            # serializes all asyncio.to_thread() calls, so only one thread
            # accesses the connection at a time.  WAL mode allows concurrent
            # readers if needed in the future.
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
            conn.commit()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            count = cursor.fetchone()[0]
            return conn, count

        # Offload blocking sqlite3 calls to a thread to avoid stalling the event loop
        self._conn, self._count = await asyncio.to_thread(_init)

    async def close(self) -> None:
        """Close the SQLite connection."""
        async with self._lock:
            self._closed = True
            if self._conn is not None:
                conn = self._conn
                self._conn = None
                await asyncio.to_thread(conn.close)

    async def append(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Insert a message row, offloading the blocking SQL write to a thread."""
        meta_json = json.dumps(metadata or {})
        sid = self._session_id

        async with self._lock:
            conn = await self._ensure_initialized()

            def _append() -> None:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                    (sid, role, content, meta_json),
                )
                conn.commit()

            await asyncio.to_thread(_append)
            self._count += 1

    async def get(self, limit: int | None = None, roles: list[str] | None = None) -> list[dict[str, Any]]:
        """Query messages, optionally filtered by role and limited to the N most recent."""
        sid = self._session_id
        safe_limit = int(limit) if limit is not None else None

        async with self._lock:
            conn = await self._ensure_initialized()

            def _get() -> list[dict[str, Any]]:
                where = "WHERE session_id = ?"
                params: list[Any] = [sid]
                if roles:
                    placeholders = ",".join("?" * len(roles))
                    where += f" AND role IN ({placeholders})"
                    params.extend(roles)
                query = f"SELECT role, content, metadata FROM messages {where}"  # nosec B608 - where clause built from hardcoded strings and parameterized placeholders only
                if safe_limit is not None:
                    query += " ORDER BY id DESC LIMIT ?"
                    params.append(safe_limit)
                else:
                    query += " ORDER BY id ASC"
                cursor = conn.execute(query, params)
                rows = [
                    {"role": row[0], "content": row[1], "metadata": json.loads(row[2])}
                    for row in cursor.fetchall()
                ]
                if safe_limit is not None:
                    rows = list(reversed(rows))
                return rows

            return await asyncio.to_thread(_get)

    async def clear(self) -> None:
        """Delete all messages for this session."""
        sid = self._session_id

        async with self._lock:
            conn = await self._ensure_initialized()

            def _clear() -> None:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.commit()

            await asyncio.to_thread(_clear)
            self._count = 0

    async def fork(self, session_id: str) -> SQLiteHistory:
        """Copy all messages into a new session via INSERT ... SELECT."""
        src_sid = self._session_id

        async with self._lock:
            conn = await self._ensure_initialized()

            def _fork() -> None:
                conn.execute(
                    """INSERT INTO messages (session_id, role, content, metadata)
                       SELECT ?, role, content, metadata FROM messages WHERE session_id = ? ORDER BY id""",
                    (session_id, src_sid),
                )
                conn.commit()

            await asyncio.to_thread(_fork)

        forked = SQLiteHistory(db_path=self._db_path, session_id=session_id)
        await forked.initialize()
        return forked

    def __len__(self) -> int:
        """Return the cached message count as a best-effort snapshot.

        This is not guaranteed to be consistent under concurrent writes
        because it reads shared state without holding the async lock.
        """
        return self._count
