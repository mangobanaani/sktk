from sktk.core.errors import (
    BlackboardTypeError,
    ContractValidationError,
    GuardrailException,
    NoCapableAgentError,
    RetryExhaustedError,
    SKTKContextError,
    SKTKError,
)


def test_sktk_error_is_base():
    err = SKTKError("base error")
    assert isinstance(err, Exception)
    assert str(err) == "base error"


def test_context_error_inherits():
    err = SKTKContextError("no context")
    assert isinstance(err, SKTKError)


def test_guardrail_exception_has_reason():
    err = GuardrailException(reason="blocked", filter_name="PII")
    assert err.reason == "blocked"
    assert err.filter_name == "PII"
    assert isinstance(err, SKTKError)


def test_blackboard_type_error_has_details():
    err = BlackboardTypeError(key="result", expected="AnalysisResult", got="str")
    assert err.key == "result"
    assert err.expected == "AnalysisResult"
    assert err.got == "str"
    assert isinstance(err, SKTKError)


def test_no_capable_agent_error():
    err = NoCapableAgentError(task_type="AnalysisRequest", available=["writer", "editor"])
    assert err.task_type == "AnalysisRequest"
    assert err.available == ["writer", "editor"]
    assert isinstance(err, SKTKError)


def test_contract_validation_error_has_raw_output():
    err = ContractValidationError(
        model_name="AnalysisResult",
        raw_output="not json",
        validation_errors=[{"loc": ["field"], "msg": "required"}],
    )
    assert err.raw_output == "not json"
    assert isinstance(err, SKTKError)


def test_retry_exhausted_error():
    err = RetryExhaustedError(attempts=3, last_error=ValueError("bad"))
    assert err.attempts == 3
    assert isinstance(err.last_error, ValueError)
    assert isinstance(err, SKTKError)
