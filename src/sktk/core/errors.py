"""SKTK exception hierarchy.

All SKTK exceptions inherit from SKTKError so callers can catch
broad or narrow as needed.
"""

from __future__ import annotations

from typing import Any


class SKTKError(Exception):
    """Base exception for all SKTK errors."""


class SKTKContextError(SKTKError):
    """Raised when execution context is missing or invalid."""


class GuardrailException(SKTKError):
    """Raised when a guardrail filter blocks execution."""

    def __init__(self, reason: str, filter_name: str) -> None:
        """Initialize with the blocking reason and the name of the filter that triggered it."""
        self.reason = reason
        self.filter_name = filter_name
        super().__init__(f"Guardrail '{filter_name}' blocked: {reason}")


class BlackboardTypeError(SKTKError):
    """Raised when a blackboard value does not match the expected type."""

    def __init__(self, key: str, expected: str, got: str) -> None:
        """Initialize with the blackboard key, expected type name, and actual type name."""
        self.key = key
        self.expected = expected
        self.got = got
        super().__init__(f"Blackboard key '{key}': expected {expected}, got {got}")


class NoCapableAgentError(SKTKError):
    """Raised when no agent can handle a given task."""

    def __init__(self, task_type: str, available: list[str]) -> None:
        """Initialize with the unhandled task type and list of available agent names."""
        self.task_type = task_type
        self.available = list(available)
        super().__init__(
            f"No agent capable of handling '{task_type}'. Available agents: {available}"
        )


class ContractValidationError(SKTKError):
    """Raised when LLM output fails to validate against an output contract.

    Note: ``raw_output`` may contain sensitive LLM-generated content.
    Avoid logging or exposing this field to untrusted consumers.
    """

    def __init__(
        self, model_name: str, raw_output: str, validation_errors: list[dict[str, Any]]
    ) -> None:
        """Initialize with target model name, raw LLM text, and Pydantic validation errors."""
        self.model_name = model_name
        self.raw_output = raw_output
        self.validation_errors = list(validation_errors)
        # Deliberately exclude raw_output from the message to avoid leaking
        # LLM-generated content into logs or error reporters.
        super().__init__(f"Failed to validate LLM output as {model_name}: {validation_errors}")


class RetryExhaustedError(SKTKError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        """Initialize with the number of attempts made and the final exception."""
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Exhausted {attempts} retries. Last error: {last_error}")


class CircuitBreakerOpenError(SKTKError, RuntimeError):
    """Raised when a call is rejected because the circuit breaker is OPEN."""

    def __init__(self, message: str = "Circuit breaker is OPEN -- calls are rejected") -> None:
        super().__init__(message)
