# sktk.testing

Mocks, fixtures, assertions, and sandbox utilities for testing SKTK agents without live LLM calls.

## Table of Contents

- [MockKernel](#mockkernel)
- [LLMScenario](#llmscenario)
- [assert_history_contains](#assert_history_contains)
- [assert_blackboard_has](#assert_blackboard_has)
- [assert_events_emitted](#assert_events_emitted)
- [mock_kernel](#mock_kernel)
- [test_session](#test_session)
- [test_blackboard](#test_blackboard)
- [SandboxResult](#sandboxresult)
- [PluginSandbox](#pluginsandbox)
- [PromptTestCase](#prompttestcase)
- [PromptTestResult](#prompttestresult)
- [PromptSuite](#promptsuite)

---

## MockKernel

Drop-in mock for the LLM kernel. Queues canned responses and records function calls for verification.

### Constructor

```python
def __init__(self) -> None
```

No parameters. Initializes empty response queue and function expectation/call lists.

### Methods

#### `expect_chat_completion`

```python
def expect_chat_completion(self, responses: list[str]) -> None
```

Enqueue one or more canned responses for subsequent chat completions.

| Param | Type | Description |
|-------|------|-------------|
| `responses` | `list[str]` | Responses to queue (consumed in FIFO order). |

#### `next_response`

```python
def next_response(self) -> str
```

Pop and return the next queued response.

**Returns:** `str`

**Raises:** `AssertionError` if no responses remain.

#### `expect_function`

```python
def expect_function(
    self,
    plugin: str,
    function: str,
    return_value: Any,
    assert_args: dict | None = None,
) -> None
```

Register an expected function call with its canned return value.

| Param | Type | Description |
|-------|------|-------------|
| `plugin` | `str` | Plugin name. |
| `function` | `str` | Function name within the plugin. |
| `return_value` | `Any` | Value to return when the function is called. |
| `assert_args` | `dict \| None` | If set, assert that these key/value pairs appear in the call args. |

#### `record_function_call`

```python
def record_function_call(self, plugin: str, function: str, args: dict) -> Any
```

Record a function call and return the matching expected value.

| Param | Type | Description |
|-------|------|-------------|
| `plugin` | `str` | Plugin name. |
| `function` | `str` | Function name. |
| `args` | `dict` | Call arguments. |

**Returns:** `Any` -- the registered return value.

**Raises:** `AssertionError` if the function call was not expected.

#### `verify`

```python
def verify(self) -> None
```

Assert that all expected responses have been consumed.

**Raises:** `AssertionError` if queued responses remain.

---

## LLMScenario

Scripted LLM response sequence supporting both successful replies and injected failures.

### Constructor

```python
def __init__(self, responses: deque[str | Exception]) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `responses` | `deque[str \| Exception]` | Ordered sequence of responses or exceptions. |

### Class Methods

#### `scripted`

```python
@classmethod
def scripted(cls, responses: list[str]) -> LLMScenario
```

Create a scenario that returns the given responses in order.

| Param | Type | Description |
|-------|------|-------------|
| `responses` | `list[str]` | Ordered list of canned responses. |

**Returns:** `LLMScenario`

#### `failing`

```python
@classmethod
def failing(cls, error: Exception, after_turns: int = 0) -> LLMScenario
```

Create a scenario that raises an error after the specified number of successful (empty) turns.

| Param | Type | Description |
|-------|------|-------------|
| `error` | `Exception` | Exception to raise. |
| `after_turns` | `int` | Number of empty-string turns before the error. Default `0`. |

**Returns:** `LLMScenario`

### Methods

#### `next`

```python
def next(self) -> str
```

Return the next scripted response. If the next item is an `Exception`, it is raised instead.

**Returns:** `str`

**Raises:** `AssertionError` if the scenario is exhausted. Raises the scripted exception if one was queued.

---

## assert_history_contains

```python
async def assert_history_contains(
    session: Session,
    role: str,
    content_pattern: str,
) -> None
```

Assert that the session history contains a message matching the given role and regex pattern.

| Param | Type | Description |
|-------|------|-------------|
| `session` | `Session` | Session whose history to search. |
| `role` | `str` | Expected message role (e.g. `"user"`, `"assistant"`). |
| `content_pattern` | `str` | Regex pattern to match against message content. |

**Raises:** `AssertionError` if no matching message is found.

---

## assert_blackboard_has

```python
async def assert_blackboard_has(
    session: Session,
    key: str,
    expected: BaseModel,
) -> None
```

Assert that the session blackboard contains the expected value at the given key.

| Param | Type | Description |
|-------|------|-------------|
| `session` | `Session` | Session whose blackboard to check. |
| `key` | `str` | Blackboard key. |
| `expected` | `BaseModel` | Expected Pydantic model value. |

**Raises:** `AssertionError` if the key is missing or the value does not match.

---

## assert_events_emitted

```python
def assert_events_emitted(events: list[Any], event_types: list[type]) -> None
```

Assert that the given event types appear in order (not necessarily contiguously) within the event list.

| Param | Type | Description |
|-------|------|-------------|
| `events` | `list[Any]` | List of emitted events. |
| `event_types` | `list[type]` | Expected event types in order. |

**Raises:** `AssertionError` if the expected sequence is not found.

---

## mock_kernel

```python
def mock_kernel() -> MockKernel
```

Factory function. Create a fresh `MockKernel` for testing without live LLM calls.

**Returns:** `MockKernel`

---

## test_session

```python
def test_session(session_id: str | None = None) -> Session
```

Factory function. Create an in-memory `Session` suitable for unit tests.

| Param | Type | Description |
|-------|------|-------------|
| `session_id` | `str \| None` | Optional session ID. Auto-generated if `None`. |

**Returns:** `Session`

---

## test_blackboard

```python
def test_blackboard() -> InMemoryBlackboard
```

Factory function. Create an empty in-memory `Blackboard` for unit tests.

**Returns:** `InMemoryBlackboard`

---

## SandboxResult

Dataclass. Result from running a tool in the `PluginSandbox`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool that was run. |
| `args` | `dict[str, Any]` | Arguments passed to the tool. |
| `output` | `Any` | Tool output (or `None` on failure). |
| `error` | `str \| None` | Error message if the tool raised. Default `None`. |
| `success` | `bool` | Whether the tool completed without error. Default `True`. |

---

## PluginSandbox

Isolated test environment for validating tools.

### Constructor

```python
def __init__(self) -> None
```

No parameters.

### Methods

#### `run`

```python
async def run(self, tool: Tool, **kwargs: Any) -> SandboxResult
```

Run a tool in the sandbox and capture its output. Exceptions are caught and stored in the result.

| Param | Type | Description |
|-------|------|-------------|
| `tool` | `Tool` | The tool to execute. |
| `**kwargs` | `Any` | Arguments forwarded to the tool. |

**Returns:** `SandboxResult`

#### `clear`

```python
def clear(self) -> None
```

Discard all stored results.

### Properties

#### `results`

```python
@property
def results(self) -> list[SandboxResult]
```

Return a copy of all sandbox results.

---

## PromptTestCase

Dataclass. A single prompt regression test case.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Test case name. |
| `prompt` | `str` | Prompt text to send. |
| `expected_contains` | `list[str]` | Substrings that must appear in the response (case-insensitive). Default `[]`. |
| `expected_not_contains` | `list[str]` | Substrings that must not appear in the response (case-insensitive). Default `[]`. |
| `max_tokens` | `int \| None` | Optional token limit hint. Default `None`. |

---

## PromptTestResult

Dataclass. Result of a prompt regression test.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `case_name` | `str` | Name of the test case. |
| `passed` | `bool` | Whether all assertions passed. |
| `response` | `str` | The raw response text. |
| `failures` | `list[str]` | List of failure descriptions. Default `[]`. |

---

## PromptSuite

Regression test suite for prompt quality. Collects `PromptTestCase` entries and runs them against an invoke function.

### Constructor

```python
def __init__(self) -> None
```

No parameters.

### Methods

#### `add_case`

```python
def add_case(self, case: PromptTestCase) -> None
```

Add a test case to the suite.

| Param | Type | Description |
|-------|------|-------------|
| `case` | `PromptTestCase` | Test case to add. |

#### `run`

```python
async def run(self, invoke_fn: Any) -> list[PromptTestResult]
```

Run all test cases against an invoke function. Each case's prompt is passed to `invoke_fn`, and the response is checked against `expected_contains` and `expected_not_contains` (case-insensitive).

| Param | Type | Description |
|-------|------|-------------|
| `invoke_fn` | `Any` | Async callable accepting a prompt string and returning a response. |

**Returns:** `list[PromptTestResult]`

### Properties

#### `cases`

```python
@property
def cases(self) -> list[PromptTestCase]
```

Return a copy of all registered test cases.
