"""Mock infrastructure for testing SKTK agents without live LLM calls."""

from __future__ import annotations

from collections import deque
from typing import Any


class MockKernel:
    """Test double for scripted chat and tool responses without live model calls."""

    def __init__(self) -> None:
        self._responses: deque[str] = deque()
        self._function_expectations: list[dict[str, Any]] = []
        self._function_calls: list[dict[str, Any]] = []

    def expect_chat_completion(self, responses: list[str]) -> None:
        """Enqueue one or more canned responses for subsequent chat completions."""
        self._responses.extend(responses)

    def next_response(self) -> str:
        """Pop and return the next queued response, or raise if empty."""
        if not self._responses:
            raise AssertionError("No more expected responses queued in MockKernel")
        return self._responses.popleft()

    def expect_function(
        self, plugin: str, function: str, return_value: Any, assert_args: dict[str, Any] | None = None
    ) -> None:
        """Register an expected function call with its canned return value."""
        self._function_expectations.append(
            {
                "plugin": plugin,
                "function": function,
                "return_value": return_value,
                "assert_args": assert_args,
            }
        )

    def record_function_call(self, plugin: str, function: str, args: dict[str, Any]) -> Any:
        """Record a function call and return the matching expected value, or fail."""
        self._function_calls.append({"plugin": plugin, "function": function, "args": args})
        for i, exp in enumerate(self._function_expectations):
            if exp["plugin"] == plugin and exp["function"] == function:
                if exp["assert_args"]:
                    for k, v in exp["assert_args"].items():
                        if args.get(k) != v:
                            raise AssertionError(
                                f"Argument mismatch for {plugin}.{function}: "
                                f"expected {k}={v!r}, got {args.get(k)!r}"
                            )
                self._function_expectations.pop(i)
                return exp["return_value"]
        raise AssertionError(f"Unexpected function call: {plugin}.{function}")

    def verify(self) -> None:
        """Assert that all expected responses and function expectations have been consumed."""
        remaining = len(self._responses)
        if remaining > 0:
            raise AssertionError(f"{remaining} expected responses not consumed in MockKernel")
        remaining_fns = len(self._function_expectations)
        if remaining_fns > 0:
            raise AssertionError(
                f"{remaining_fns} expected function calls not consumed in MockKernel"
            )


class LLMScenario:
    """Scenario helper that emits deterministic responses or exceptions per turn."""

    def __init__(self, responses: deque[str | Exception]) -> None:
        self._responses = responses

    @classmethod
    def scripted(cls, responses: list[str]) -> LLMScenario:
        """Create a scenario that returns the given responses in order."""
        return cls(deque(responses))

    @classmethod
    def failing(cls, error: Exception, after_turns: int = 0) -> LLMScenario:
        """Create a scenario that raises an error after the specified number of turns."""
        items: deque[str | Exception] = deque()
        for i in range(after_turns):
            items.append(f"[placeholder response {i + 1}]")
        items.append(error)
        return cls(items)

    def next(self) -> str:
        """Return the next scripted response, or raise if it is an exception."""
        if not self._responses:
            raise AssertionError("LLMScenario exhausted")
        item = self._responses.popleft()
        if isinstance(item, Exception):
            raise item
        return item
