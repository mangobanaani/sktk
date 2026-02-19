"""Convenience builders for agents with router + default safety filters."""

from __future__ import annotations

from typing import Iterable

from sktk.agent.agent import SKTKAgent
from sktk.agent.capabilities import Capability
from sktk.agent.filters import ContentSafetyFilter, PIIFilter, PromptInjectionFilter
from sktk.agent.router import Router
from sktk.agent.tools import Tool
from sktk.session.session import Session

DEFAULT_BLOCKED_PATTERNS = [
    r"(?i)password",
    r"(?i)api[-_ ]?key",
]


def default_safety_filters(blocked_patterns: Iterable[str] | None = None):
    """Return the default guardrail filter stack for safe agents."""

    patterns = list(DEFAULT_BLOCKED_PATTERNS if blocked_patterns is None else blocked_patterns)
    return [
        PromptInjectionFilter(),
        PIIFilter(),
        ContentSafetyFilter(blocked_patterns=patterns),
    ]


def build_safe_agent(
    name: str,
    instructions: str,
    router: Router,
    *,
    session: Session | None = None,
    tools: list[Tool] | None = None,
    capabilities: list[Capability] | None = None,
    blocked_patterns: Iterable[str] | None = None,
    instructions_version: str | None = None,
) -> SKTKAgent:
    """Create an agent pre-wired with router and default safety filters.

    - Uses the router to complete; tools/capabilities optional.
    - Safety filters: prompt injection, PII, content safety (configurable).
    """

    filters = default_safety_filters(blocked_patterns)

    agent = SKTKAgent(
        name=name,
        instructions=instructions,
        instructions_version=instructions_version,
        session=session,
        filters=filters,
        tools=tools or [],
        capabilities=capabilities or [],
        service=router,
    )
    return agent
