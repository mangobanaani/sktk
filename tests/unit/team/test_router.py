# tests/unit/team/test_router.py
import pytest
from pydantic import BaseModel

from sktk.agent.agent import SKTKAgent
from sktk.agent.capabilities import Capability
from sktk.core.errors import NoCapableAgentError
from sktk.team.router import CapabilityRouter


class QueryRequest(BaseModel):
    query: str


class ReportRequest(BaseModel):
    topic: str


@pytest.fixture
def router():
    analyst = SKTKAgent(
        name="analyst",
        instructions="Analyze.",
        capabilities=[
            Capability(
                name="analysis",
                description="Analyze data",
                input_types=[QueryRequest],
                output_types=[],
            )
        ],
        input_contract=QueryRequest,
    )
    writer = SKTKAgent(
        name="writer",
        instructions="Write.",
        capabilities=[
            Capability(
                name="reporting",
                description="Write reports",
                input_types=[ReportRequest],
                output_types=[],
            )
        ],
        input_contract=ReportRequest,
    )
    return CapabilityRouter(agents=[analyst, writer])


def test_route_by_input_type(router):
    agent = router.route(input_type=QueryRequest)
    assert agent.name == "analyst"


def test_route_by_input_type_report(router):
    agent = router.route(input_type=ReportRequest)
    assert agent.name == "writer"


def test_route_no_match_raises(router):
    class UnknownRequest(BaseModel):
        x: str

    with pytest.raises(NoCapableAgentError) as exc_info:
        router.route(input_type=UnknownRequest)
    assert "UnknownRequest" in str(exc_info.value)
    assert "analyst" in exc_info.value.available
    assert "writer" in exc_info.value.available


def test_route_by_tags():
    agent = SKTKAgent(
        name="tagged",
        instructions="T.",
        capabilities=[
            Capability(name="x", description="", input_types=[], output_types=[], tags=["finance"])
        ],
    )
    router = CapabilityRouter(agents=[agent])
    result = router.route(tags=["finance"])
    assert result.name == "tagged"


def test_route_skips_agents_without_capabilities():
    agent_no_caps = SKTKAgent(name="empty", instructions="No caps.")
    agent_with_caps = SKTKAgent(
        name="capable",
        instructions="Has caps.",
        capabilities=[
            Capability(name="x", description="", input_types=[QueryRequest], output_types=[])
        ],
    )
    router = CapabilityRouter(agents=[agent_no_caps, agent_with_caps])
    result = router.route(input_type=QueryRequest)
    assert result.name == "capable"
