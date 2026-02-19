# tests/unit/agent/test_capabilities.py
from pydantic import BaseModel

from sktk.agent.capabilities import Capability, match_capabilities


class AnalysisRequest(BaseModel):
    query: str


class AnalysisResult(BaseModel):
    answer: str


class ReportRequest(BaseModel):
    topic: str


def test_capability_creation():
    cap = Capability(
        name="data_analysis",
        description="Analyze structured data",
        input_types=[AnalysisRequest],
        output_types=[AnalysisResult],
        tags=["analysis", "data"],
    )
    assert cap.name == "data_analysis"
    assert len(cap.input_types) == 1
    assert cap.tags == ["analysis", "data"]


def test_capability_default_tags():
    cap = Capability(
        name="writing",
        description="Write content",
        input_types=[],
        output_types=[],
    )
    assert cap.tags == []


def test_match_capabilities_exact_type():
    caps = [
        Capability(
            name="analysis",
            description="Analyze data",
            input_types=[AnalysisRequest],
            output_types=[AnalysisResult],
        ),
        Capability(
            name="reporting",
            description="Write reports",
            input_types=[ReportRequest],
            output_types=[],
        ),
    ]
    matches = match_capabilities(caps, input_type=AnalysisRequest)
    assert len(matches) == 1
    assert matches[0].name == "analysis"


def test_match_capabilities_by_tag():
    caps = [
        Capability(name="a", description="", input_types=[], output_types=[], tags=["finance"]),
        Capability(name="b", description="", input_types=[], output_types=[], tags=["tech"]),
    ]
    matches = match_capabilities(caps, tags=["finance"])
    assert len(matches) == 1
    assert matches[0].name == "a"


def test_match_capabilities_no_match():
    caps = [
        Capability(name="a", description="", input_types=[AnalysisRequest], output_types=[]),
    ]
    matches = match_capabilities(caps, input_type=ReportRequest)
    assert len(matches) == 0
