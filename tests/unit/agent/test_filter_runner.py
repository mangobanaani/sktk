# tests/unit/agent/test_filter_runner.py
import pytest

from sktk.agent.filter_runner import FilterRunner
from sktk.agent.filters import FilterAdapter, FilterContext
from sktk.core.errors import GuardrailException
from sktk.core.types import Deny, FilterResult, Modify


class AllowFilter(FilterAdapter):
    """Filter that allows everything (default FilterAdapter behaviour)."""


class DenyFilter(FilterAdapter):
    """Filter that denies everything."""

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Deny(reason="blocked by test")

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Deny(reason="blocked by test")


class ModifyFilter(FilterAdapter):
    """Filter that uppercases content."""

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Modify(content=context.content.upper())

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Modify(content=context.content.upper())


def test_active_returns_false_when_no_filters():
    runner = FilterRunner(filters=[], agent_name="a")
    assert runner.active is False


def test_active_returns_true_when_filters_exist():
    runner = FilterRunner(filters=[AllowFilter()], agent_name="a")
    assert runner.active is True


@pytest.mark.asyncio
async def test_run_input_allow_passes_content_through():
    runner = FilterRunner(filters=[AllowFilter()], agent_name="a")
    result = await runner.run_input("hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_run_input_deny_raises_guardrail_exception():
    runner = FilterRunner(filters=[DenyFilter()], agent_name="a")
    with pytest.raises(GuardrailException, match="blocked by test"):
        await runner.run_input("hello")


@pytest.mark.asyncio
async def test_run_input_modify_changes_content():
    runner = FilterRunner(filters=[ModifyFilter()], agent_name="a")
    result = await runner.run_input("hello")
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_run_output_allow_passes_through():
    runner = FilterRunner(filters=[AllowFilter()], agent_name="a")
    result = await runner.run_output("world")
    assert result == "world"


@pytest.mark.asyncio
async def test_run_output_deny_raises_guardrail_exception():
    runner = FilterRunner(filters=[DenyFilter()], agent_name="a")
    with pytest.raises(GuardrailException, match="blocked by test"):
        await runner.run_output("world")
