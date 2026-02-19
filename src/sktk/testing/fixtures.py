"""Reusable pytest fixtures for testing SKTK components."""

from __future__ import annotations

import uuid

from sktk.session.backends.memory import InMemoryBlackboard, InMemoryHistory
from sktk.session.session import Session
from sktk.testing.mocks import MockKernel


def mock_kernel() -> MockKernel:
    """Create a fresh MockKernel for testing without live LLM calls."""
    return MockKernel()


def test_session(session_id: str | None = None) -> Session:
    """Create an in-memory Session suitable for unit tests."""
    return Session(
        id=session_id or f"test-{uuid.uuid4().hex[:8]}",
        history=InMemoryHistory(),
        blackboard=InMemoryBlackboard(),
    )


def test_blackboard() -> InMemoryBlackboard:
    """Create an empty in-memory Blackboard for unit tests."""
    return InMemoryBlackboard()
