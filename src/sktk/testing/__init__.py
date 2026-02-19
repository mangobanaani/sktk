"""SKTK testing infrastructure -- mocks, fixtures, assertions, sandbox."""

from sktk.testing.assertions import (
    assert_blackboard_has,
    assert_events_emitted,
    assert_history_contains,
)
from sktk.testing.fixtures import mock_kernel, test_blackboard, test_session
from sktk.testing.mocks import LLMScenario, MockKernel
from sktk.testing.sandbox import (
    PluginSandbox,
    PromptSuite,
    PromptTestCase,
    PromptTestResult,
    SandboxResult,
)

__all__ = [
    "LLMScenario",
    "MockKernel",
    "PluginSandbox",
    "PromptSuite",
    "PromptTestCase",
    "PromptTestResult",
    "SandboxResult",
    "assert_blackboard_has",
    "assert_events_emitted",
    "assert_history_contains",
    "mock_kernel",
    "test_blackboard",
    "test_session",
]
