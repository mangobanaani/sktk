"""MCP server support -- expose agents as MCP tool servers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sktk.agent.agent import SKTKAgent

logger = logging.getLogger(__name__)


def expose_as_mcp_server(agent: SKTKAgent) -> Any:
    """Create an MCP server that exposes an agent's tools and invoke().

    Usage:
        from sktk.agent.mcp_server import expose_as_mcp_server
        server = expose_as_mcp_server(agent)
        # Run server with appropriate transport
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool
    except ImportError as exc:
        raise ImportError(
            "MCP server support requires the 'mcp' package: pip install sktk[mcp]"
        ) from exc

    server = Server(agent.name)

    @server.list_tools()  # type: ignore[misc, no-untyped-call]
    async def list_tools() -> list[MCPTool]:
        tools = []
        # Expose agent tools
        for tool in getattr(agent, "tools", []):
            tools.append(
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.parameters or {"type": "object", "properties": {}},
                )
            )
        # Expose invoke as a tool
        tools.append(
            MCPTool(
                name="invoke",
                description=f"Send a message to the {agent.name} agent",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message to send"}
                    },
                    "required": ["message"],
                },
            )
        )
        return tools

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
        arguments = arguments or {}
        try:
            if name == "invoke":
                message = arguments.get("message", "")
                result = await agent.invoke(message)
                return [TextContent(type="text", text=str(result))]
            # Call agent tool
            result = await agent.call_tool(name, **arguments)
            return [TextContent(type="text", text=str(result))]
        except KeyError:
            logger.warning("MCP call_tool: unknown tool '%s'", name)
            return [TextContent(type="text", text=f"Error: unknown tool '{name}'")]
        except Exception as exc:
            logger.error("MCP call_tool %s failed: %s", name, exc, exc_info=True)
            return [TextContent(type="text", text="Error: tool execution failed")]

    return server
