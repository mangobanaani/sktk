# tests/unit/agent/test_contracts.py
import pytest
from pydantic import BaseModel

from sktk.agent.contracts import parse_output, serialize_input
from sktk.core.errors import ContractValidationError


class AnalysisRequest(BaseModel):
    symbol: str
    period: str


class AnalysisResult(BaseModel):
    summary: str
    score: float


def test_serialize_input_to_prompt():
    req = AnalysisRequest(symbol="AAPL", period="Q3 2024")
    prompt = serialize_input(req)
    assert "AAPL" in prompt
    assert "Q3 2024" in prompt


def test_serialize_input_custom_template():
    req = AnalysisRequest(symbol="AAPL", period="Q3 2024")
    template = "Analyze {symbol} for {period}"
    prompt = serialize_input(req, template=template)
    assert prompt == "Analyze AAPL for Q3 2024"


def test_parse_output_valid_json():
    raw = '{"summary": "Strong quarter", "score": 0.85}'
    result = parse_output(raw, AnalysisResult)
    assert result.summary == "Strong quarter"
    assert result.score == 0.85


def test_parse_output_json_in_markdown():
    raw = 'Here is the result:\n```json\n{"summary": "Good", "score": 0.9}\n```'
    result = parse_output(raw, AnalysisResult)
    assert result.summary == "Good"


def test_parse_output_invalid_raises():
    raw = "This is not JSON at all"
    with pytest.raises(ContractValidationError) as exc_info:
        parse_output(raw, AnalysisResult)
    assert exc_info.value.raw_output == raw
    assert exc_info.value.model_name == "AnalysisResult"


def test_parse_output_partial_json_raises():
    raw = '{"summary": "Missing score field"}'
    with pytest.raises(ContractValidationError):
        parse_output(raw, AnalysisResult)


def test_parse_output_json_embedded_in_text():
    raw = 'Here is the result: {"summary": "Embedded", "score": 0.7} and some trailing text'
    result = parse_output(raw, AnalysisResult)
    assert result.summary == "Embedded"
    assert result.score == 0.7


def test_parse_output_markdown_no_json_tag():
    raw = '```\n{"summary": "No tag", "score": 0.6}\n```'
    result = parse_output(raw, AnalysisResult)
    assert result.summary == "No tag"


def test_parse_output_invalid_markdown_and_embedded_json_paths_raise():
    raw = "```json\n{this is not valid json}\n``` trailing {also invalid}"
    with pytest.raises(ContractValidationError):
        parse_output(raw, AnalysisResult)
