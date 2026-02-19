"""Custom assertion helpers for testing SKTK components."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from sktk.session.session import Session


async def assert_history_contains(session: Session, role: str, content_pattern: str) -> None:
    """Assert that the session history contains a message matching role and regex pattern."""
    messages = await session.history.get()
    for msg in messages:
        if msg["role"] == role and re.search(content_pattern, msg["content"]):
            return
    raise AssertionError(
        f"No message found with role='{role}' matching '{content_pattern}'. Messages: {messages}"
    )


async def assert_blackboard_has(session: Session, key: str, expected: BaseModel) -> None:
    """Assert that the blackboard contains the expected value at the given key."""
    got = await session.blackboard.get(key, type(expected))
    if got is None:
        raise AssertionError(f"Blackboard key '{key}' not found")
    if got != expected:
        raise AssertionError(f"Blackboard key '{key}': expected {expected}, got {got}")


def assert_events_emitted(events: list[Any], event_types: list[type]) -> None:
    """Assert that the given event types appear in order within the event list."""
    type_iter = iter(event_types)
    current = next(type_iter, None)
    for event in events:
        if current is not None and isinstance(event, current):
            current = next(type_iter, None)
    if current is not None:
        remaining = [current] + list(type_iter)
        names = [t.__name__ for t in remaining]
        raise AssertionError(f"Events not found in sequence: {names}")
