"""A2A (Agent-to-Agent) protocol support.

Implements basic A2A protocol for inter-agent communication over HTTP,
following the A2A specification for agent discovery and invocation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentCard:
    """A2A Agent Card describing an agent's capabilities and endpoint."""

    name: str
    description: str
    endpoint: str
    capabilities: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.endpoint,
            "capabilities": self.capabilities,
            "defaultInputModes": self.input_modes,
            "defaultOutputModes": self.output_modes,
        }


class A2AClient:
    """Client for discovering and invoking remote A2A agents.

    Usage:
        client = A2AClient()
        card = await client.discover("http://remote-agent:8080")
        result = await client.invoke(card, "Hello, agent!")
    """

    def __init__(self) -> None:
        self._http: Any = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _get_client(self) -> Any:
        async with self._lock:
            if self._http is None:
                try:
                    import httpx
                except ImportError as exc:
                    raise ImportError("A2A support requires httpx: pip install sktk[a2a]") from exc
                self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def discover(self, base_url: str) -> AgentCard:
        """Discover a remote agent by fetching its agent card."""
        client = await self._get_client()
        url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        return AgentCard(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            endpoint=data.get("url", base_url),
            capabilities=data.get("capabilities", []),
            input_modes=data.get("defaultInputModes", ["text/plain"]),
            output_modes=data.get("defaultOutputModes", ["text/plain"]),
        )

    async def invoke(self, card: AgentCard, message: str) -> str:
        """Invoke a remote A2A agent with a text message."""
        client = await self._get_client()
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "tasks/send",
            "params": {
                "id": uuid.uuid4().hex,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
        }
        response = await client.post(card.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        # Extract text from response
        result = data.get("result", {})
        artifacts = result.get("artifacts", [])
        if artifacts:
            parts = artifacts[0].get("parts", [])
            if parts:
                return str(parts[0].get("text", str(data)))
        return str(data)

    async def close(self) -> None:
        async with self._lock:
            if self._http is not None:
                await self._http.aclose()
                self._http = None


class A2AServer:
    """Expose an sktk agent as an A2A-compatible HTTP endpoint.

    Usage:
        server = A2AServer(agent, port=8080)
        response = await server.handle_request(request_body)
    """

    def __init__(self, agent: Any, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._agent = agent
        self._host = host
        self._port = port

    def agent_card(self) -> AgentCard:
        """Generate the agent card for this agent."""
        capabilities = [cap.name for cap in getattr(self._agent, "capabilities", [])]
        instructions = getattr(self._agent, "instructions", "") or ""
        return AgentCard(
            name=self._agent.name,
            description=instructions[:200],
            endpoint=f"http://{self._host}:{self._port}",
            capabilities=capabilities,
        )

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Build a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    async def handle_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming A2A JSON-RPC request."""
        request_id = body.get("id")
        if body.get("jsonrpc") != "2.0":
            return self._error_response(
                request_id,
                -32600,
                "Invalid Request: missing or incorrect jsonrpc version",
            )
        try:
            method = body.get("method", "")
            params = body.get("params", {})

            if not isinstance(params, dict):
                return self._error_response(request_id, -32602, "Invalid params: expected object")

            if method == "tasks/send":
                message = params.get("message", {})
                if not isinstance(message, dict):
                    return self._error_response(
                        request_id, -32602, "Invalid params: message must be object"
                    )
                parts = message.get("parts", [])
                if not isinstance(parts, list):
                    return self._error_response(
                        request_id, -32602, "Invalid params: parts must be array"
                    )
                text = parts[0].get("text", "") if parts and isinstance(parts[0], dict) else ""
                result = await self._agent.invoke(text)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "id": params.get("id", ""),
                        "status": {"state": "completed"},
                        "artifacts": [
                            {
                                "parts": [{"type": "text", "text": str(result)}],
                            }
                        ],
                    },
                }

            return self._error_response(request_id, -32601, f"Method not found: {method}")

        except Exception:
            logger.exception("A2A request handling failed")
            return self._error_response(request_id, -32603, "Internal error")
