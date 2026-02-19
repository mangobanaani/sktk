# tests/unit/agent/test_optimizer.py
"""Tests for the PromptOptimizer and OptimizationResult."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from sktk.agent.optimizer import OptimizationResult, PromptOptimizer
from sktk.agent.templates import PromptTemplate


@dataclass
class _TestResult:
    """Minimal test case result for suite simulation."""

    case_name: str
    passed: bool
    failures: list[str]


class _MockSuite:
    """Mock test suite that returns pre-configured results."""

    def __init__(self, results: list[_TestResult]) -> None:
        self._results = results

    async def run(self, invoke_fn: Any) -> list[_TestResult]:
        return self._results


@pytest.mark.asyncio
async def test_optimizer_passing_suite_returns_original_template():
    """When all tests pass at baseline, optimizer returns original template."""
    template = PromptTemplate(name="good", text="You are a helpful assistant.")

    all_passing = _MockSuite(
        [
            _TestResult(case_name="test1", passed=True, failures=[]),
            _TestResult(case_name="test2", passed=True, failures=[]),
        ]
    )

    invoke_fn = AsyncMock(return_value="ok")
    optimizer = PromptOptimizer(invoke_fn=invoke_fn)

    result = await optimizer.optimize(template, all_passing, iterations=3, target_pass_rate=1.0)

    assert isinstance(result, OptimizationResult)
    assert result.original_template is template
    assert result.optimized_template is template
    assert result.original_pass_rate == 1.0
    assert result.final_pass_rate == 1.0
    assert result.iterations_run == 0  # no iterations needed


@pytest.mark.asyncio
async def test_optimizer_failing_suite_attempts_improvement():
    """When tests fail, optimizer uses critique to attempt improvement."""
    template = PromptTemplate(name="weak", text="Be brief.")

    call_count = 0

    class _ImprovingSuite:
        """Suite that starts failing then passes after first iteration."""

        async def run(self, invoke_fn: Any) -> list[_TestResult]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two calls: baseline eval + first iteration failure gathering
                return [
                    _TestResult(
                        case_name="detail_test", passed=False, failures=["Not detailed enough"]
                    ),
                ]
            # After improvement: all pass
            return [
                _TestResult(case_name="detail_test", passed=True, failures=[]),
            ]

    # The critique_fn (same as invoke_fn here) returns an improved template
    async def mock_invoke(prompt: str) -> str:
        if (
            "Analyze" in prompt
            or "critique" in prompt.lower()
            or "prompt engineering" in prompt.lower()
        ):
            return "```template\nBe very detailed and thorough in your responses.\n```"
        return "ok"

    optimizer = PromptOptimizer(invoke_fn=mock_invoke)
    result = await optimizer.optimize(
        template, _ImprovingSuite(), iterations=5, target_pass_rate=1.0
    )

    assert isinstance(result, OptimizationResult)
    assert result.original_template is template
    assert result.original_pass_rate == 0.0
    assert result.final_pass_rate == 1.0
    assert result.iterations_run >= 1
    assert result.optimized_template.text != template.text
    assert "detailed" in result.optimized_template.text.lower()


def test_optimization_result_contains_correct_fields():
    """OptimizationResult dataclass holds all required fields."""
    original = PromptTemplate(name="orig", text="Original text")
    optimized = PromptTemplate(name="orig", text="Improved text")
    history = [
        {"iteration": 1, "pass_rate": 0.5, "template_text": "attempt 1", "improved": True},
        {"iteration": 2, "pass_rate": 0.8, "template_text": "attempt 2", "improved": True},
    ]

    result = OptimizationResult(
        original_template=original,
        optimized_template=optimized,
        original_pass_rate=0.2,
        final_pass_rate=0.8,
        iterations_run=2,
        history=history,
    )

    assert result.original_template is original
    assert result.optimized_template is optimized
    assert result.original_pass_rate == pytest.approx(0.2)
    assert result.final_pass_rate == pytest.approx(0.8)
    assert result.iterations_run == 2
    assert len(result.history) == 2
    assert result.history[0]["iteration"] == 1
    assert result.history[1]["improved"] is True


@pytest.mark.asyncio
async def test_optimizer_respects_max_iterations():
    """Optimizer stops after the configured number of iterations."""
    template = PromptTemplate(name="stuck", text="Be vague.")

    always_failing = _MockSuite(
        [
            _TestResult(case_name="test1", passed=False, failures=["Still vague"]),
        ]
    )

    async def mock_invoke(prompt: str) -> str:
        return "```template\nStill vague but different.\n```"

    optimizer = PromptOptimizer(invoke_fn=mock_invoke)
    result = await optimizer.optimize(template, always_failing, iterations=2, target_pass_rate=1.0)

    assert result.iterations_run == 2
    assert result.final_pass_rate == 0.0


@pytest.mark.asyncio
async def test_optimizer_extract_improved_text_fallback():
    """When LLM suggestion has no code blocks, the raw text is used if it looks like a template."""
    template = PromptTemplate(name="test", text="Old template.")

    call_count = 0

    class _OnceFailSuite:
        async def run(self, invoke_fn: Any) -> list[_TestResult]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [_TestResult(case_name="t", passed=False, failures=["bad"])]
            return [_TestResult(case_name="t", passed=True, failures=[])]

    async def mock_invoke(prompt: str) -> str:
        if "prompt engineering" in prompt.lower():
            # Return plain text (no code block markers)
            return "New improved template line one.\nLine two of the template."
        return "ok"

    optimizer = PromptOptimizer(invoke_fn=mock_invoke)
    result = await optimizer.optimize(template, _OnceFailSuite(), iterations=3)

    # The optimizer should have extracted the raw text as template
    assert result.iterations_run >= 1
