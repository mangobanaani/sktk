# tests/unit/agent/test_a2a.py
"""Tests for A2A (Agent-to-Agent) protocol support."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sktk.agent.a2a import A2AClient, A2AServer, AgentCard

# ---------------------------------------------------------------
# AgentCard tests
# ---------------------------------------------------------------


def test_agent_card_creation_with_correct_fields():
    """AgentCard stores all fields and converts to dict correctly."""
    card = AgentCard(
        name="researcher",
        description="Research agent",
        endpoint="http://localhost:8080",
        capabilities=["search", "summarize"],
        input_modes=["text/plain", "application/json"],
        output_modes=["text/plain"],
    )
    assert card.name == "researcher"
    assert card.description == "Research agent"
    assert card.endpoint == "http://localhost:8080"
    assert card.capabilities == ["search", "summarize"]

    d = card.to_dict()
    assert d["name"] == "researcher"
    assert d["description"] == "Research agent"
    assert d["url"] == "http://localhost:8080"
    assert d["capabilities"] == ["search", "summarize"]
    assert d["defaultInputModes"] == ["text/plain", "application/json"]
    assert d["defaultOutputModes"] == ["text/plain"]


def test_agent_card_default_modes():
    """AgentCard defaults to text/plain for input and output modes."""
    card = AgentCard(name="bot", description="A bot", endpoint="http://x")
    assert card.input_modes == ["text/plain"]
    assert card.output_modes == ["text/plain"]


# ---------------------------------------------------------------
# A2AClient tests
# ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a2a_client_discover_fetches_and_parses_agent_card():
    """A2AClient.discover fetches /.well-known/agent.json and parses it."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "name": "remote-agent",
        "description": "A remote agent",
        "url": "http://remote:9090",
        "capabilities": ["chat"],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    client = A2AClient()
    client._http = mock_http

    card = await client.discover("http://remote:9090")

    mock_http.get.assert_awaited_once_with("http://remote:9090/.well-known/agent.json")
    assert card.name == "remote-agent"
    assert card.description == "A remote agent"
    assert card.endpoint == "http://remote:9090"
    assert card.capabilities == ["chat"]


@pytest.mark.asyncio
async def test_a2a_client_invoke_sends_correct_json_rpc_request():
    """A2AClient.invoke sends a JSON-RPC tasks/send request."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp1",
        "result": {
            "artifacts": [{"parts": [{"type": "text", "text": "Hello from remote"}]}],
        },
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    client = A2AClient()
    client._http = mock_http

    card = AgentCard(
        name="remote",
        description="Remote agent",
        endpoint="http://remote:9090/a2a",
    )
    result = await client.invoke(card, "Hello")

    assert result == "Hello from remote"
    # Verify the POST was made to the card endpoint
    call_args = mock_http.post.call_args
    assert call_args[0][0] == "http://remote:9090/a2a"
    payload = call_args[1]["json"]
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "tasks/send"
    assert payload["params"]["message"]["role"] == "user"
    assert payload["params"]["message"]["parts"][0]["text"] == "Hello"


@pytest.mark.asyncio
async def test_a2a_client_invoke_no_artifacts_returns_str():
    """When response has no artifacts, return stringified data."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp2",
        "result": {},
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    client = A2AClient()
    client._http = mock_http

    card = AgentCard(name="r", description="", endpoint="http://x")
    result = await client.invoke(card, "test")
    # Falls back to str(data)
    assert isinstance(result, str)


# ---------------------------------------------------------------
# A2AServer tests
# ---------------------------------------------------------------


def test_a2a_server_agent_card_has_correct_fields():
    """A2AServer.agent_card() creates correct well-known endpoint data."""
    cap = SimpleNamespace(name="summarize")
    agent = SimpleNamespace(
        name="my-agent",
        instructions="I summarize things for you.",
        capabilities=[cap],
    )
    server = A2AServer(agent, host="0.0.0.0", port=9090)
    card = server.agent_card()

    assert card.name == "my-agent"
    assert card.description == "I summarize things for you."
    assert card.endpoint == "http://0.0.0.0:9090"
    assert card.capabilities == ["summarize"]


@pytest.mark.asyncio
async def test_a2a_server_handle_request_tasks_send():
    """A2AServer.handle_request delegates tasks/send to agent.invoke."""
    agent = AsyncMock()
    agent.name = "test-agent"
    agent.invoke = AsyncMock(return_value="response text")

    server = A2AServer(agent)
    body = {
        "jsonrpc": "2.0",
        "id": "req1",
        "method": "tasks/send",
        "params": {
            "id": "task1",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello agent"}],
            },
        },
    }
    result = await server.handle_request(body)

    agent.invoke.assert_awaited_once_with("Hello agent")
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "req1"
    assert result["result"]["status"]["state"] == "completed"
    assert result["result"]["artifacts"][0]["parts"][0]["text"] == "response text"


@pytest.mark.asyncio
async def test_a2a_server_handle_request_unknown_method():
    """A2AServer returns error for unknown JSON-RPC methods."""
    agent = AsyncMock()
    agent.name = "test-agent"

    server = A2AServer(agent)
    body = {
        "jsonrpc": "2.0",
        "id": "req2",
        "method": "unknown/method",
        "params": {},
    }
    result = await server.handle_request(body)

    assert result["error"]["code"] == -32601
    assert "unknown/method" in result["error"]["message"]
