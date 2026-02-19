"""Prompt optimization via iterative hill-climbing on test suites.

Lightweight prompt improvement: run a suite against a template,
use an LLM to critique and refine, repeat until pass rate improves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from sktk.agent.templates import PromptTemplate

logger = logging.getLogger(__name__)

InvokeFn = Callable[[str], Awaitable[Any]]


@runtime_checkable
class TestCaseResult(Protocol):
    """Protocol for individual test case results returned by a TestSuite."""

    @property
    def passed(self) -> bool: ...

    @property
    def case_name(self) -> str: ...

    @property
    def failures(self) -> list[str]: ...


@runtime_checkable
class TestSuite(Protocol):
    """Protocol for test suites used by the prompt optimizer.

    A test suite accepts an invocation function and returns a list of
    test case results, each having ``passed``, ``case_name``, and
    ``failures`` attributes.
    """

    async def run(self, invoke_fn: InvokeFn) -> list[TestCaseResult]: ...


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""

    original_template: PromptTemplate
    optimized_template: PromptTemplate
    original_pass_rate: float
    final_pass_rate: float
    iterations_run: int
    history: list[dict[str, Any]] = field(default_factory=list)


class PromptOptimizer:
    """Iteratively improve a prompt template by running a test suite.

    Strategies:
    - Self-critique: ask the LLM to analyze failures and suggest improvements
    - Few-shot selection: include examples of passing cases as demonstrations

    Usage:
        optimizer = PromptOptimizer(invoke_fn=agent.invoke)
        result = await optimizer.optimize(
            template=my_template,
            suite=my_suite,
            iterations=5,
        )
        improved = result.optimized_template
    """

    def __init__(
        self,
        invoke_fn: InvokeFn,
        critique_fn: InvokeFn | None = None,
    ) -> None:
        self._invoke = invoke_fn
        self._critique = critique_fn or invoke_fn

    def _make_invoke_fn(self, template: PromptTemplate) -> InvokeFn:
        """Create an invoke function that prepends the rendered template."""

        async def invoke_fn(prompt: str) -> Any:
            return await self._invoke(template.render(**template.defaults) + "\n\n" + prompt)

        return invoke_fn

    async def optimize(
        self,
        template: PromptTemplate,
        suite: TestSuite,
        iterations: int = 5,
        target_pass_rate: float = 1.0,
    ) -> OptimizationResult:
        """Run iterative optimization on a template using the test suite.

        Note: The template must have all variables covered by defaults,
        as the optimizer renders the template with defaults only.
        Parameterized templates that require additional variables at
        render time are not supported.
        """
        current = template
        history: list[dict[str, Any]] = []

        # Evaluate baseline
        baseline_rate = await self._evaluate(current, suite)
        best_rate = baseline_rate
        best_template = current

        for i in range(iterations):
            if best_rate >= target_pass_rate:
                break

            # Get failures
            results = await suite.run(self._make_invoke_fn(current))
            failures = [r for r in results if not r.passed]
            if not failures:
                break

            # Ask LLM to critique and suggest improvements
            critique_prompt = self._build_critique_prompt(current, failures)
            suggestion = str(await self._critique(critique_prompt))

            # Extract improved template from suggestion
            improved_text = self._extract_improved_text(suggestion, current.text)
            candidate = PromptTemplate(
                name=current.name,
                text=improved_text,
                defaults=current.defaults,
            )

            # Evaluate candidate
            candidate_rate = await self._evaluate(candidate, suite)

            entry = {
                "iteration": i + 1,
                "pass_rate": candidate_rate,
                "template_text": improved_text[:200],
                "improved": candidate_rate > best_rate,
            }
            history.append(entry)
            logger.info(
                "Iteration %d: pass_rate=%.2f (best=%.2f)",
                i + 1,
                candidate_rate,
                best_rate,
            )

            # Hill-climbing: only keep if better
            if candidate_rate > best_rate:
                best_rate = candidate_rate
                best_template = candidate
                current = candidate

        return OptimizationResult(
            original_template=template,
            optimized_template=best_template,
            original_pass_rate=baseline_rate,
            final_pass_rate=best_rate,
            iterations_run=len(history),
            history=history,
        )

    async def _evaluate(self, template: PromptTemplate, suite: TestSuite) -> float:
        """Run the suite and return pass rate."""
        results = await suite.run(self._make_invoke_fn(template))
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return passed / len(results)

    def _build_critique_prompt(
        self, template: PromptTemplate, failures: list[TestCaseResult]
    ) -> str:
        """Build a prompt asking the LLM to critique and improve the template."""
        failure_descriptions = []
        for f in failures[:5]:  # Limit to 5 failures
            desc = f"- Case '{f.case_name}': {'; '.join(f.failures)}"
            failure_descriptions.append(desc)

        return (
            "You are a prompt engineering expert. Analyze the following prompt template "
            "and its test failures, then provide an improved version.\n\n"
            f"Current template:\n```\n{template.text}\n```\n\n"
            f"Failures:\n" + "\n".join(failure_descriptions) + "\n\n"
            "Provide an improved template that addresses these failures. "
            "Return ONLY the improved template text, wrapped in ```template``` markers."
        )

    def _extract_improved_text(self, suggestion: str, fallback: str) -> str:
        """Extract improved template text from LLM suggestion."""
        import re

        # Try ```template``` first
        match = re.search(r"```template\s*\n?(.*?)\n?```", suggestion, re.DOTALL)
        if not match:
            # Fall back to untagged code blocks
            match = re.search(r"```\s*\n(.*?)\n```", suggestion, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If the whole suggestion looks like a template (no markdown), use it
        stripped = suggestion.strip()
        lines = stripped.split("\n")
        if len(lines) >= 2 and not stripped.startswith("#"):
            # If the original template has variables, the improvement should too
            original_vars = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", fallback))
            if original_vars:
                suggestion_vars = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", stripped))
                if not suggestion_vars:
                    return (
                        fallback  # LLM response has no template variables — likely not a template
                    )
            return stripped

        return fallback
