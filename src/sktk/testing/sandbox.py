"""Plugin sandbox and prompt regression testing.

PluginSandbox: isolated test environment for tools.
PromptSuite: regression test suite for prompt quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sktk.agent.tools import Tool


@dataclass
class SandboxResult:
    """Result from running a tool in the sandbox."""

    tool_name: str
    args: dict[str, Any]
    output: Any
    error: str | None = None
    success: bool = True


class PluginSandbox:
    """Isolated test environment for validating tools.

    Usage:
        sandbox = PluginSandbox()
        result = await sandbox.run(my_tool, query="test")
        assert result.success
        assert "expected" in str(result.output)
    """

    def __init__(self) -> None:
        self._results: list[SandboxResult] = []

    async def run(self, tool: Tool, **kwargs: Any) -> SandboxResult:
        """Run a tool in the sandbox and capture its output."""
        try:
            output = await tool(**kwargs)
            result = SandboxResult(
                tool_name=tool.name,
                args=kwargs,
                output=output,
            )
        except Exception as e:
            result = SandboxResult(
                tool_name=tool.name,
                args=kwargs,
                output=None,
                error=str(e),
                success=False,
            )
        self._results.append(result)
        return result

    @property
    def results(self) -> list[SandboxResult]:
        return list(self._results)

    def clear(self) -> None:
        self._results.clear()


@dataclass
class PromptTestCase:
    """A single prompt regression test case."""

    name: str
    prompt: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    max_tokens: int | None = None


@dataclass
class PromptTestResult:
    """Result of a prompt regression test."""

    case_name: str
    passed: bool
    response: str
    failures: list[str] = field(default_factory=list)


class PromptSuite:
    """Regression test suite for prompt quality.

    Usage:
        suite = PromptSuite()
        suite.add_case(PromptTestCase(
            name="greeting",
            prompt="Say hello",
            expected_contains=["hello", "Hi"],
        ))

        results = await suite.run(agent.invoke)
        assert all(r.passed for r in results)
    """

    def __init__(self) -> None:
        self._cases: list[PromptTestCase] = []

    def add_case(self, case: PromptTestCase) -> None:
        self._cases.append(case)

    @property
    def cases(self) -> list[PromptTestCase]:
        return list(self._cases)

    async def run(self, invoke_fn: Any) -> list[PromptTestResult]:
        """Run all test cases against an invoke function."""
        results = []
        for case in self._cases:
            response = str(await invoke_fn(case.prompt))
            failures = []

            for expected in case.expected_contains:
                if expected.lower() not in response.lower():
                    failures.append(f"Expected '{expected}' not found in response")

            for not_expected in case.expected_not_contains:
                if not_expected.lower() in response.lower():
                    failures.append(f"Unexpected '{not_expected}' found in response")

            results.append(
                PromptTestResult(
                    case_name=case.name,
                    passed=len(failures) == 0,
                    response=response,
                    failures=failures,
                )
            )
        return results
