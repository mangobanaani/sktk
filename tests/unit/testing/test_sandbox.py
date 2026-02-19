# tests/unit/testing/test_sandbox.py
import pytest

from sktk.agent.tools import tool
from sktk.testing.sandbox import PluginSandbox, PromptSuite, PromptTestCase


@pytest.mark.asyncio
async def test_sandbox_run_success():
    @tool(description="Add numbers")
    def add(a: int, b: int) -> int:
        return a + b

    sandbox = PluginSandbox()
    result = await sandbox.run(add, a=2, b=3)
    assert result.success
    assert result.output == 5
    assert result.error is None


@pytest.mark.asyncio
async def test_sandbox_run_error():
    @tool(description="Fail")
    def fail() -> None:
        raise ValueError("boom")

    sandbox = PluginSandbox()
    result = await sandbox.run(fail)
    assert not result.success
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_sandbox_results():
    @tool(description="Echo")
    def echo(msg: str) -> str:
        return msg

    sandbox = PluginSandbox()
    await sandbox.run(echo, msg="hello")
    await sandbox.run(echo, msg="world")
    assert len(sandbox.results) == 2


def test_sandbox_clear():
    sandbox = PluginSandbox()
    sandbox._results.append(None)
    sandbox.clear()
    assert len(sandbox.results) == 0


@pytest.mark.asyncio
async def test_prompt_suite_pass():
    async def invoke(prompt):
        return "Hello World! Nice to meet you."

    suite = PromptSuite()
    suite.add_case(
        PromptTestCase(
            name="greeting",
            prompt="Say hello",
            expected_contains=["hello"],
        )
    )
    results = await suite.run(invoke)
    assert len(results) == 1
    assert results[0].passed


@pytest.mark.asyncio
async def test_prompt_suite_fail():
    async def invoke(prompt):
        return "Goodbye."

    suite = PromptSuite()
    suite.add_case(
        PromptTestCase(
            name="greeting",
            prompt="Say hello",
            expected_contains=["hello"],
        )
    )
    results = await suite.run(invoke)
    assert not results[0].passed
    assert len(results[0].failures) == 1


@pytest.mark.asyncio
async def test_prompt_suite_not_contains():
    async def invoke(prompt):
        return "Here is a safe response."

    suite = PromptSuite()
    suite.add_case(
        PromptTestCase(
            name="safety",
            prompt="Test",
            expected_not_contains=["unsafe", "dangerous"],
        )
    )
    results = await suite.run(invoke)
    assert results[0].passed


@pytest.mark.asyncio
async def test_prompt_suite_not_contains_failure():
    async def invoke(prompt):
        return "This response is unsafe."

    suite = PromptSuite()
    suite.add_case(
        PromptTestCase(
            name="safety_fail",
            prompt="Test",
            expected_not_contains=["unsafe"],
        )
    )
    results = await suite.run(invoke)
    assert not results[0].passed
    assert "Unexpected 'unsafe'" in results[0].failures[0]


@pytest.mark.asyncio
async def test_prompt_suite_cases():
    suite = PromptSuite()
    suite.add_case(PromptTestCase(name="a", prompt="test"))
    suite.add_case(PromptTestCase(name="b", prompt="test2"))
    assert len(suite.cases) == 2
