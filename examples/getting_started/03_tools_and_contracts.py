"""Tools and typed contracts.

Shows how to register callable tools on an agent using the @tool
decorator, and how to enforce typed input/output contracts with
Pydantic models so agents speak a structured schema.

Usage:
    python examples/getting_started/03_tools_and_contracts.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _provider import get_provider
from pydantic import BaseModel

from sktk import SKTKAgent, tool

# -- Tools: functions the agent can call during conversation --


@tool(description="Add two numbers together")
async def add(a: int, b: int) -> int:
    return a + b


@tool(description="Multiply two numbers together")
async def multiply(a: int, b: int) -> int:
    return a * b


# -- Contracts: Pydantic models that enforce structured I/O --


class MathRequest(BaseModel):
    """What the agent expects as input."""

    expression: str


class MathResponse(BaseModel):
    """What the agent must return."""

    answer: str
    confidence: float


async def main() -> None:
    provider = get_provider()

    # 1) Inspect tool schemas (what the LLM sees for function-calling)
    print("=== Tool Schemas ===")
    for t in [add, multiply]:
        print(f"  {t.name}: {t.to_schema()}")

    # 2) Call tools directly (no LLM needed)
    print(f"\n  add(3, 4)       = {await add(a=3, b=4)}")
    print(f"  multiply(5, 6)  = {await multiply(a=5, b=6)}")

    # 3) Register tools on an agent
    agent = SKTKAgent(
        name="math-agent",
        instructions=(
            "You are a math assistant. When asked a math question, respond with "
            'ONLY a JSON object like {"answer": "42", "confidence": 0.99}. '
            "No other text."
        ),
        service=provider,
        timeout=30.0,
        tools=[add, multiply],
        output_contract=MathResponse,
    )

    # 4) Look up a tool by name
    found = agent.get_tool("add")
    print("\n=== Tool Lookup ===")
    print(f"  agent.get_tool('add') -> {found}")

    # 5) Invoke with typed output contract -- raw JSON is parsed into MathResponse
    result = await agent.invoke("What is 6 * 7?")
    print("\n=== Typed Output ===")
    print(f"  type   : {type(result).__name__}")
    print(f"  answer : {result.answer}")
    print(f"  conf.  : {result.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
