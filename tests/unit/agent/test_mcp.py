# tests/unit/agent/test_mcp.py
"""Tests for MCP client (MCPToolProvider) and server (expose_as_mcp_server)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------
# MCPToolProvider tests
# ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_provider_tools_returns_tool_objects():
    """MCPToolProvider.tools() returns Tool objects from discovered MCP tools."""
    from sktk.agent.mcp import MCPToolProvider

    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._transport = None
    provider._client = None

    # Simulate discovered tools by directly populating _tools
    from sktk.agent.tools import Tool

    fake_tool_def = SimpleNamespace(
        name="search",
        description="Search the web",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    provider._tools = [provider._wrap_tool(fake_tool_def)]

    tools = provider.tools()
    assert len(tools) == 1
    assert isinstance(tools[0], Tool)
    assert tools[0].name == "search"
    assert tools[0].description == "Search the web"
    assert tools[0].parameters == {"type": "object", "properties": {"q": {"type": "string"}}}


@pytest.mark.asyncio
async def test_mcp_tool_provider_wrap_tool_calls_mcp():
    """Wrapped MCP tool delegates to client.call_tool with correct arguments."""
    from sktk.agent.mcp import MCPToolProvider

    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._transport = None

    # Mock the MCP client
    text_block = SimpleNamespace(text="result text")
    call_result = SimpleNamespace(content=[text_block])
    provider._client = AsyncMock()
    provider._client.call_tool = AsyncMock(return_value=call_result)

    fake_tool_def = SimpleNamespace(
        name="lookup",
        description="Look up data",
        inputSchema={},
    )
    tool = provider._wrap_tool(fake_tool_def)
    result = await tool(key="value")

    provider._client.call_tool.assert_awaited_once_with("lookup", arguments={"key": "value"})
    assert result == "result text"


@pytest.mark.asyncio
async def test_mcp_tool_provider_wrap_tool_not_connected_raises():
    """Calling a wrapped tool when provider is not connected raises RuntimeError."""
    from sktk.agent.mcp import MCPToolProvider

    provider = MCPToolProvider.__new__(MCPToolProvider)
    provider._transport = None
    provider._client = None

    fake_tool_def = SimpleNamespace(name="broken", description="", inputSchema={})
    tool = provider._wrap_tool(fake_tool_def)

    with pytest.raises(RuntimeError, match="not connected"):
        await tool()


@pytest.mark.asyncio
async def test_mcp_tool_provider_connect_import_error():
    """MCPToolProvider.connect() raises ImportError when mcp package is missing."""
    from sktk.agent.mcp import MCPToolProvider

    provider = MCPToolProvider(transport=("r", "w"))

    with patch.dict(sys.modules, {"mcp": None}), pytest.raises(ImportError, match="mcp"):
        await provider.connect()


# ---------------------------------------------------------------
# expose_as_mcp_server tests
# ---------------------------------------------------------------


def test_expose_as_mcp_server_import_error():
    """expose_as_mcp_server raises ImportError when mcp package is missing."""
    from sktk.agent.mcp_server import expose_as_mcp_server

    agent = SimpleNamespace(name="test-agent", tools=[])

    with (
        patch.dict(sys.modules, {"mcp": None, "mcp.server": None, "mcp.types": None}),
        pytest.raises(ImportError, match="mcp"),
    ):
        expose_as_mcp_server(agent)


@pytest.mark.asyncio
async def test_expose_as_mcp_server_creates_server_with_tools():
    """expose_as_mcp_server creates a server exposing agent tools + invoke."""
    from sktk.agent.tools import Tool

    def dummy_fn():
        return "ok"

    agent_tool = Tool(name="greet", description="Say hello", fn=dummy_fn, parameters={})
    agent = SimpleNamespace(name="test-agent", tools=[agent_tool])

    # Build a minimal mock Server that captures registered handlers
    registered_handlers = {}

    class MockServer:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            def decorator(fn):
                registered_handlers["list_tools"] = fn
                return fn

            return decorator

        def call_tool(self):
            def decorator(fn):
                registered_handlers["call_tool"] = fn
                return fn

            return decorator

    mock_server_module = MagicMock()
    mock_server_module.Server = MockServer

    mock_types_module = MagicMock()
    mock_types_module.Tool = lambda **kw: SimpleNamespace(**kw)
    mock_types_module.TextContent = lambda **kw: SimpleNamespace(**kw)

    with patch.dict(
        sys.modules,
        {
            "mcp": MagicMock(),
            "mcp.server": mock_server_module,
            "mcp.types": mock_types_module,
        },
    ):
        from importlib import reload

        import sktk.agent.mcp_server as mcp_server_mod

        reload(mcp_server_mod)
        server = mcp_server_mod.expose_as_mcp_server(agent)

    assert isinstance(server, MockServer)
    assert server.name == "test-agent"

    # list_tools should include agent tools + invoke
    tools = await registered_handlers["list_tools"]()
    tool_names = [t.name for t in tools]
    assert "greet" in tool_names
    assert "invoke" in tool_names


@pytest.mark.asyncio
async def test_expose_as_mcp_server_call_tool_delegates_to_agent():
    """MCP server call_tool delegates to agent.invoke and agent.call_tool."""
    agent = AsyncMock()
    agent.name = "test-agent"
    agent.tools = []
    agent.invoke = AsyncMock(return_value="agent response")
    agent.call_tool = AsyncMock(return_value="tool result")

    registered_handlers = {}

    class MockServer:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            def decorator(fn):
                registered_handlers["list_tools"] = fn
                return fn

            return decorator

        def call_tool(self):
            def decorator(fn):
                registered_handlers["call_tool"] = fn
                return fn

            return decorator

    mock_server_module = MagicMock()
    mock_server_module.Server = MockServer

    mock_types_module = MagicMock()
    mock_types_module.Tool = lambda **kw: SimpleNamespace(**kw)
    mock_types_module.TextContent = lambda **kw: SimpleNamespace(**kw)

    with patch.dict(
        sys.modules,
        {
            "mcp": MagicMock(),
            "mcp.server": mock_server_module,
            "mcp.types": mock_types_module,
        },
    ):
        from importlib import reload

        import sktk.agent.mcp_server as mcp_server_mod

        reload(mcp_server_mod)
        mcp_server_mod.expose_as_mcp_server(agent)

    # Test invoke path
    result = await registered_handlers["call_tool"]("invoke", {"message": "hello"})
    agent.invoke.assert_awaited_once_with("hello")
    assert result[0].text == "agent response"

    # Test tool call path
    result = await registered_handlers["call_tool"]("my_tool", {"arg": "val"})
    agent.call_tool.assert_awaited_once_with("my_tool", arg="val")
    assert result[0].text == "tool result"
