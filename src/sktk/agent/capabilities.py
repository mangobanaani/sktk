"""Agent capability declarations and matching."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class Capability:
    """Structured declaration of what an agent can do."""

    name: str
    description: str
    input_types: list[type[BaseModel]]
    output_types: list[type[BaseModel]]
    tags: list[str] = field(default_factory=list)


def match_capabilities(
    capabilities: list[Capability],
    *,
    input_type: type[BaseModel] | None = None,
    tags: list[str] | None = None,
) -> list[Capability]:
    """Find capabilities matching input type and/or tags."""
    results = capabilities

    if input_type is not None:
        results = [c for c in results if input_type in c.input_types]

    if tags is not None:
        tag_set = set(tags)
        results = [c for c in results if tag_set & set(c.tags)]

    return results
