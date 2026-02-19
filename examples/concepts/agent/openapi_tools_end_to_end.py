"""OpenAPI tool generation end-to-end.

Shows how to turn an OpenAPI spec into callable tools, register them on an
agent, and invoke both tool stubs and normal agent responses.

Usage:
    python examples/concepts/agent/openapi_tools_end_to_end.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _provider import get_provider

from sktk import SKTKAgent, tools_from_openapi

PETSTORE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Demo API", "version": "1.0"},
    "paths": {
        "/pets": {
            "get": {
                "operationId": "list_pets",
                "summary": "List pets",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer"},
                    }
                ],
            },
            "post": {
                "operationId": "create_pet",
                "summary": "Create pet",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "pet_name": {
                                        "type": "string",
                                        "description": "Pet name",
                                    },
                                    "tag": {"type": "string"},
                                },
                                "required": ["pet_name"],
                            }
                        }
                    }
                },
            },
        }
    },
}


async def main() -> None:
    provider = get_provider()
    tools = tools_from_openapi(PETSTORE_SPEC)

    print("Generated tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    agent = SKTKAgent(
        name="api-agent",
        instructions="Use available API tools before answering. Summarize what you did.",
        service=provider,
        timeout=30.0,
        tools=tools,
    )

    # The generated tools are safe stubs by default. Replace `status` with real HTTP logic
    # in production integrations.
    list_result = await agent.call_tool("list_pets", limit=5)
    create_result = await agent.call_tool("create_pet", pet_name="milo", tag="cat")

    print("\nTool stub outputs:")
    print(f"  list_pets  -> {list_result}")
    print(f"  create_pet -> {create_result}")

    answer = await agent.invoke("Create a pet payload and summarize what you did.")
    print("\nAgent response:")
    print(f"  {answer}")


if __name__ == "__main__":
    asyncio.run(main())
