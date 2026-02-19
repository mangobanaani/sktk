"""Shared fixtures for integration tests that hit the real Claude API."""

from __future__ import annotations

import os

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.providers import AnthropicClaudeProvider
from sktk.agent.router import FallbackPolicy, Router
from sktk.core.secrets import FileSecretsProvider
from sktk.session.session import Session

_FLAG_VALUES = {"1", "true", "yes"}


def _integration_enabled() -> bool:
    return os.environ.get("SKTK_RUN_INTEGRATION", "").lower() in _FLAG_VALUES


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _integration_enabled():
        return
    for item in items:
        if not item.nodeid.startswith("tests/integration/"):
            continue
        if "test_flag_gate.py" in item.nodeid:
            continue
        item.add_marker(pytest.mark.skip(reason="SKTK_RUN_INTEGRATION not enabled"))

MODEL = os.environ.get("SKTK_TEST_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 256


def api_key() -> str:
    if not _integration_enabled():
        pytest.skip("SKTK_RUN_INTEGRATION not enabled")
    secrets = FileSecretsProvider(".env")
    key = secrets.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not found in .env")
    return key


@pytest.fixture(scope="session", name="api_key")
def api_key_fixture() -> str:
    return api_key()


@pytest.fixture(scope="session")
def anthropic_client(api_key: str):
    try:
        import anthropic
    except ImportError:
        pytest.skip("anthropic SDK not installed")
    return anthropic.AsyncAnthropic(api_key=api_key)


@pytest.fixture()
def claude_provider(anthropic_client) -> AnthropicClaudeProvider:
    return AnthropicClaudeProvider(
        client=anthropic_client,
        model=MODEL,
        max_tokens=MAX_TOKENS,
    )


@pytest.fixture()
def claude_agent(claude_provider) -> SKTKAgent:
    return SKTKAgent(
        name="test-agent",
        instructions="You are a helpful assistant. Be concise.",
        service=claude_provider,
        timeout=30.0,
    )


@pytest.fixture()
def claude_router(claude_provider) -> Router:
    return Router(providers=[claude_provider], policy=FallbackPolicy())


@pytest.fixture()
def session() -> Session:
    return Session(id="integration-test")
