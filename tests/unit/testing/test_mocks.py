import pytest

from sktk.testing.mocks import LLMScenario, MockKernel


def test_mock_kernel_creation():
    mk = MockKernel()
    assert mk is not None


def test_mock_kernel_expect_chat_completion():
    mk = MockKernel()
    mk.expect_chat_completion(responses=["Hello!", "Goodbye!"])
    assert mk.next_response() == "Hello!"
    assert mk.next_response() == "Goodbye!"


def test_mock_kernel_exhausted_raises():
    mk = MockKernel()
    mk.expect_chat_completion(responses=["only one"])
    mk.next_response()
    with pytest.raises(AssertionError, match="No more expected responses"):
        mk.next_response()


def test_mock_kernel_verify_all_consumed():
    mk = MockKernel()
    mk.expect_chat_completion(responses=["a", "b"])
    mk.next_response()
    with pytest.raises(AssertionError, match="1 expected responses not consumed"):
        mk.verify()


def test_mock_kernel_verify_passes_when_all_consumed():
    mk = MockKernel()
    mk.expect_chat_completion(responses=["a"])
    mk.next_response()
    mk.verify()


def test_llm_scenario_scripted():
    scenario = LLMScenario.scripted(["response 1", "response 2"])
    assert scenario.next() == "response 1"
    assert scenario.next() == "response 2"


def test_llm_scenario_failing():
    scenario = LLMScenario.failing(ValueError("boom"), after_turns=1)
    assert scenario.next() == "[placeholder response 1]"
    with pytest.raises(ValueError, match="boom"):
        scenario.next()


def test_llm_scenario_exhausted():
    scenario = LLMScenario.scripted(["one"])
    scenario.next()
    with pytest.raises(AssertionError, match="exhausted"):
        scenario.next()


def test_mock_kernel_expect_function():
    mk = MockKernel()
    mk.expect_function(plugin="math", function="add", return_value=42, assert_args={"a": 1, "b": 2})
    result = mk.record_function_call("math", "add", {"a": 1, "b": 2})
    assert result == 42


def test_mock_kernel_unexpected_function():
    mk = MockKernel()
    with pytest.raises(AssertionError, match="Unexpected function call"):
        mk.record_function_call("unknown", "func", {})


def test_mock_kernel_function_no_assert_args():
    mk = MockKernel()
    mk.expect_function(plugin="p", function="f", return_value="ok")
    result = mk.record_function_call("p", "f", {"any": "args"})
    assert result == "ok"
