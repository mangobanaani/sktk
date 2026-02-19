"""Integration tests for Tools alongside real Claude API."""

from __future__ import annotations

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.tools import Tool, tool

pytestmark = pytest.mark.integration


def _add(a: int, b: int) -> int:
    return a + b


add_tool = Tool(
    name="add",
    description="Add two integers",
    fn=_add,
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
)


async def test_tool_invocation_alongside_llm(claude_provider):
    agent = SKTKAgent(
        name="tool-agent",
        instructions="Be concise.",
        tools=[add_tool],
        service=claude_provider,
        timeout=30.0,
    )
    tool_result = await agent.call_tool("add", a=3, b=4)
    assert tool_result == 7
    # LLM still works
    llm_result = await agent.invoke("Say ok")
    assert isinstance(llm_result, str)
    assert len(llm_result) > 0


async def test_async_tool_invocation(claude_provider):
    @tool(name="async_greet", description="Greet someone")
    async def async_greet(person: str) -> str:
        return f"Hello, {person}!"

    agent = SKTKAgent(
        name="async-tool-agent",
        instructions="Be concise.",
        tools=[async_greet],
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.call_tool("async_greet", person="World")
    assert result == "Hello, World!"


async def test_tool_schema_generation():
    schema = add_tool.to_schema()
    assert schema["name"] == "add"
    assert "description" in schema
    assert "parameters" in schema
    params = schema["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
