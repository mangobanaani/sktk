"""Integration tests for input/output contracts with real Claude API."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sktk.agent.agent import SKTKAgent

pytestmark = pytest.mark.integration


class MathResult(BaseModel):
    answer: int


class CityInfo(BaseModel):
    city: str
    country: str


class MathQuestion(BaseModel):
    operation: str
    a: int
    b: int


async def test_output_contract_parses_json(claude_provider):
    agent = SKTKAgent(
        name="math-agent",
        instructions=(
            "You are a math assistant. Always respond with valid JSON "
            'matching this schema: {"answer": <integer>}. '
            "No other text, just JSON."
        ),
        output_contract=MathResult,
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("What is 7 * 8?")
    assert isinstance(result, MathResult)
    assert result.answer == 56


async def test_output_contract_markdown_json(claude_provider):
    agent = SKTKAgent(
        name="city-agent",
        instructions=(
            "You are a geography assistant. Always respond with JSON in a "
            "markdown code block like: ```json\\n{...}\\n```. "
            'Use this schema: {"city": "<name>", "country": "<name>"}.'
        ),
        output_contract=CityInfo,
        service=claude_provider,
        timeout=30.0,
    )
    result = await agent.invoke("Tell me about Paris.")
    assert isinstance(result, CityInfo)
    assert "paris" in result.city.lower()
    assert "france" in result.country.lower()


async def test_input_contract_serialization(claude_provider):
    agent = SKTKAgent(
        name="echo-agent",
        instructions=(
            "You receive structured input. Echo back the operation, a, and b values "
            "in a short sentence."
        ),
        input_contract=MathQuestion,
        service=claude_provider,
        timeout=30.0,
    )
    question = MathQuestion(operation="multiply", a=6, b=9)
    result = await agent.invoke(question)
    assert isinstance(result, str)
    assert "6" in result
    assert "9" in result
