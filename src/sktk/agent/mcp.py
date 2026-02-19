"""MCP client support -- consume tools from MCP servers."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from sktk.agent.tools import Tool

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """Connect to an MCP server and discover its tools.

    Wraps each MCP tool as an sktk Tool object so agents can consume them.

    Usage:
        provider = MCPToolProvider(transport=(read_stream, write_stream))
        await provider.connect()
        agent = SKTKAgent(name="mcp-agent", tools=provider.tools())
    """

    def __init__(self, transport: Any = None) -> None:
        self._transport = transport
        self._tools: list[Tool] = []
        self._client: Any = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    async def connect(self) -> None:
        """Connect to the MCP server and discover available tools."""
        if self._client is not None:
            return

        try:
            from mcp import ClientSession
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the 'mcp' package: pip install sktk[mcp]"
            ) from exc

        if self._transport is None:
            raise ValueError("MCPToolProvider requires a transport (read, write) tuple")

        read, write = self._transport
        session = ClientSession(read, write)
        stack = contextlib.AsyncExitStack()
        try:
            self._client = await stack.enter_async_context(session)
            await self._client.initialize()
            self._exit_stack = stack
        except Exception:
            await stack.aclose()
            self._client = None
            raise

        tools_response = await self._client.list_tools()
        self._tools = []
        for tool_def in tools_response.tools:
            self._tools.append(self._wrap_tool(tool_def))
        logger.info("Discovered %d MCP tools", len(self._tools))

    def _wrap_tool(self, tool_def: Any) -> Tool:
        """Wrap an MCP tool definition as an sktk Tool."""
        name = tool_def.name
        description = getattr(tool_def, "description", "") or ""
        raw_schema = getattr(tool_def, "inputSchema", None)
        parameters = raw_schema if isinstance(raw_schema, dict) else {}

        async def call_mcp_tool(**kwargs: Any) -> Any:
            if self._client is None:
                raise RuntimeError("MCPToolProvider not connected")
            result = await self._client.call_tool(name, arguments=kwargs)
            # MCP returns content blocks; extract text
            if hasattr(result, "content") and isinstance(result.content, list):
                texts = [getattr(block, "text", str(block)) for block in result.content]
                return "\n".join(texts)
            return str(result)

        call_mcp_tool.__name__ = name
        return Tool(
            name=name,
            description=description,
            fn=call_mcp_tool,
            parameters=parameters,
        )

    def tools(self) -> list[Tool]:
        """Return discovered tools as sktk Tool objects."""
        return list(self._tools)

    async def __aenter__(self) -> MCPToolProvider:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Disconnect from the MCP server."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._client = None
