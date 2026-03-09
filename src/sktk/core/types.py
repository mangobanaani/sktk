"""Shared types and Pydantic models used across SKTK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, TypeVar, overload

from pydantic import BaseModel, computed_field

T = TypeVar("T")

AgentName = str
SessionId = str
CorrelationId = str


class TokenUsage(BaseModel):
    """Token consumption for a single LLM call or aggregated across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two TokenUsage instances. Cost is preserved if either side has it."""
        cost = None
        if self.total_cost_usd is not None or other.total_cost_usd is not None:
            cost = (self.total_cost_usd or 0.0) + (other.total_cost_usd or 0.0)
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_cost_usd=cost,
        )


@dataclass(frozen=True)
class Allow:
    """Filter passes — continue execution."""

    allowed: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class Deny:
    """Filter blocks — stop execution."""

    reason: str
    allowed: bool = False


@dataclass(frozen=True)
class Modify:
    """Filter passes with modified content."""

    content: str
    allowed: bool = True


FilterResult = Allow | Deny | Modify


@overload
async def maybe_await(value: Awaitable[T]) -> T: ...
@overload
async def maybe_await(value: T) -> T: ...


async def maybe_await(value):  # type: ignore[no-untyped-def]
    """Await value if it's awaitable, otherwise return as-is."""
    if hasattr(value, "__await__"):
        return await value
    return value
