# sktk.core

Context propagation, events, errors, types, resilience, secrets, and configuration for SKTK.

---

## Table of Contents

- [Context](#context)
  - [ExecutionContext](#executioncontext)
  - [context_scope](#context_scope)
  - [get_context](#get_context)
  - [require_context](#require_context)
  - [propagate_context](#propagate_context)
- [Events](#events)
  - [ThinkingEvent](#thinkingevent)
  - [ToolCallEvent](#toolcallevent)
  - [RetrievalEvent](#retrievalevent)
  - [MessageEvent](#messageevent)
  - [CompletionEvent](#completionevent)
- [Types](#types)
  - [TokenUsage](#tokenusage)
  - [Allow](#allow)
  - [Deny](#deny)
  - [Modify](#modify)
  - [FilterResult](#filterresult)
  - [Type Aliases](#type-aliases)
- [Errors](#errors)
  - [SKTKError](#sktkerror)
  - [SKTKContextError](#sktkcontexterror)
  - [GuardrailException](#guardrailexception)
  - [BlackboardTypeError](#blackboardtypeerror)
  - [NoCapableAgentError](#nocapableagenterror)
  - [ContractValidationError](#contractvalidationerror)
  - [RetryExhaustedError](#retryexhaustederror)
- [Resilience](#resilience)
  - [BackoffStrategy](#backoffstrategy)
  - [RetryPolicy](#retrypolicy)
  - [CircuitState](#circuitstate)
  - [CircuitBreaker](#circuitbreaker)
- [Executor](#executor)
  - [run_in_executor](#run_in_executor)
  - [run_parallel](#run_parallel)
- [Secrets](#secrets)
  - [SecretsProvider](#secretsprovider)
  - [EnvSecretsProvider](#envsecretsprovider)
  - [FileSecretsProvider](#filesecretsprovider)
  - [ChainedSecretsProvider](#chainedsecretsprovider)
- [Config](#config)
  - [SKTKConfig](#sktkconfig)
  - [ModelConfig](#modelconfig)
  - [RetryConfig](#retryconfig)
  - [LoggingConfig](#loggingconfig)

---

## Context

Defined in `sktk.core.context`. Execution context propagation via `contextvars`.

### ExecutionContext

Immutable execution context carried through async call chains.

```python
@dataclass(frozen=True)
class ExecutionContext
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `correlation_id` | `str` | Unique identifier correlating related operations. |
| `tenant_id` | `str \| None` | Optional tenant identifier. |
| `user_id` | `str \| None` | Optional user identifier. |
| `session_id` | `str \| None` | Optional session identifier. |
| `parent_correlation_id` | `str \| None` | Correlation ID of the parent context, if any. |
| `metadata` | `dict[str, str]` | Arbitrary key-value metadata. Defaults to `{}`. |

---

### context_scope

Async context manager that sets an execution context for the duration of the block and restores the previous context on exit.

```python
@asynccontextmanager
async def context_scope(
    ctx: ExecutionContext | None = None,
    **kwargs: Any
) -> AsyncIterator[ExecutionContext]
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `ExecutionContext \| None` | An existing context to set. If `None`, a new context is created from `kwargs`. |
| `**kwargs` | `Any` | Fields forwarded to `ExecutionContext()` when `ctx` is `None`. Accepts `correlation_id`, `tenant_id`, `user_id`, `session_id`, `parent_correlation_id`, `metadata`. A random UUID is generated for `correlation_id` if omitted. |

**Returns:** `AsyncIterator[ExecutionContext]` -- yields the active `ExecutionContext`.

---

### get_context

Return the current execution context, or `None` if not set.

```python
def get_context() -> ExecutionContext | None
```

**Returns:** `ExecutionContext | None`

---

### require_context

Return the current execution context. Raises `SKTKContextError` if no context is set.

```python
def require_context() -> ExecutionContext
```

**Returns:** `ExecutionContext`

**Raises:** `SKTKContextError` -- if no execution context is active.

---

### propagate_context

Decorator ensuring the current execution context propagates into asyncio tasks.

```python
def propagate_context(
    fn: Callable[P, Awaitable[R]]
) -> Callable[P, Awaitable[R]]
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `fn` | `Callable[P, Awaitable[R]]` | The async function to wrap. |

**Returns:** `Callable[P, Awaitable[R]]` -- wrapped function that preserves context across `asyncio` task boundaries.

---

## Events

Defined in `sktk.core.events`. Typed frozen dataclasses emitted during SKTK execution. Each event exposes a `kind` property returning a string tag.

### ThinkingEvent

Emitted when an agent begins a thinking/reasoning step.

```python
@dataclass(frozen=True)
class ThinkingEvent
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `agent` | `str` | Name of the agent. |
| `correlation_id` | `str` | Correlation ID for this execution. |
| `timestamp` | `datetime` | When the event occurred. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `kind` | `str` | Returns `"thinking"`. |

---

### ToolCallEvent

Emitted when an agent invokes a plugin function.

```python
@dataclass(frozen=True)
class ToolCallEvent
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `agent` | `str` | Name of the agent. |
| `plugin` | `str` | Plugin name. |
| `function` | `str` | Function name within the plugin. |
| `arguments` | `dict[str, Any]` | Arguments passed to the function. |
| `correlation_id` | `str` | Correlation ID for this execution. |
| `timestamp` | `datetime` | When the event occurred. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `kind` | `str` | Returns `"tool_call"`. |

---

### RetrievalEvent

Emitted when an agent retrieves chunks from a knowledge base.

```python
@dataclass(frozen=True)
class RetrievalEvent
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `agent` | `str` | Name of the agent. |
| `query` | `str` | The retrieval query. |
| `chunks_retrieved` | `int` | Number of chunks returned. |
| `top_score` | `float` | Similarity score of the top result. |
| `correlation_id` | `str` | Correlation ID for this execution. |
| `timestamp` | `datetime` | When the event occurred. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `kind` | `str` | Returns `"retrieval"`. |

---

### MessageEvent

Emitted when the LLM produces a message response.

```python
@dataclass(frozen=True)
class MessageEvent
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `agent` | `str` | Name of the agent. |
| `role` | `str` | Message role (e.g. `"assistant"`). |
| `content` | `str` | Message content. |
| `token_usage` | `TokenUsage` | Token consumption for this message. |
| `correlation_id` | `str` | Correlation ID for this execution. |
| `timestamp` | `datetime` | When the event occurred. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `kind` | `str` | Returns `"message"`. |

---

### CompletionEvent

Emitted when an agent run finishes, carrying aggregated stats.

```python
@dataclass(frozen=True)
class CompletionEvent
```

**Fields**

| Field | Type | Description |
|---|---|---|
| `result` | `Any` | The final result of the agent run. |
| `total_rounds` | `int` | Number of LLM rounds executed. |
| `total_tokens` | `TokenUsage` | Aggregated token usage. |
| `duration_seconds` | `float` | Wall-clock duration. |
| `correlation_id` | `str` | Correlation ID for this execution. |
| `timestamp` | `datetime` | When the event occurred. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `kind` | `str` | Returns `"completion"`. |

---

## Types

Defined in `sktk.core.types`. Shared types and Pydantic models used across SKTK.

### TokenUsage

Token consumption for a single LLM call or aggregated across calls. Pydantic `BaseModel`.

```python
class TokenUsage(BaseModel)
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt_tokens` | `int` | `0` | Number of prompt tokens consumed. |
| `completion_tokens` | `int` | `0` | Number of completion tokens consumed. |
| `total_cost_usd` | `float \| None` | `None` | Estimated total cost in USD. |

**Computed Fields**

| Property | Type | Description |
|---|---|---|
| `total_tokens` | `int` | Sum of `prompt_tokens` and `completion_tokens`. |

**Methods**

```python
def __add__(self, other: TokenUsage) -> TokenUsage
```

Sum two `TokenUsage` instances. Cost is preserved when at least one side has it; summed when both sides have it.

---

### Allow

Filter result indicating execution should continue.

```python
@dataclass(frozen=True)
class Allow
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `allowed` | `bool` | `True` | Always `True`. |
| `reason` | `str \| None` | `None` | Optional reason for allowing. |

---

### Deny

Filter result indicating execution should be blocked.

```python
@dataclass(frozen=True)
class Deny
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `reason` | `str` | *(required)* | Reason for blocking. |
| `allowed` | `bool` | `False` | Always `False`. |

---

### Modify

Filter result indicating execution continues with modified content.

```python
@dataclass(frozen=True)
class Modify
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | *(required)* | The modified content. |
| `allowed` | `bool` | `True` | Always `True`. |

---

### FilterResult

Union type for filter outcomes.

```python
FilterResult = Allow | Deny | Modify
```

---

### Type Aliases

```python
AgentName = str
SessionId = str
CorrelationId = str
```

---

## Errors

Defined in `sktk.core.errors`. All exceptions inherit from `SKTKError`.

### SKTKError

```python
class SKTKError(Exception)
```

Base exception for all SKTK errors. Callers can catch this to handle any SKTK-specific error.

---

### SKTKContextError

```python
class SKTKContextError(SKTKError)
```

Raised when execution context is missing or invalid.

---

### GuardrailException

```python
class GuardrailException(SKTKError)
```

Raised when a guardrail filter blocks execution.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `reason` | `str` | The blocking reason. |
| `filter_name` | `str` | Name of the filter that triggered the block. |

**Attributes:** `reason`, `filter_name`

---

### BlackboardTypeError

```python
class BlackboardTypeError(SKTKError)
```

Raised when a blackboard value does not match the expected type.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `key` | `str` | The blackboard key. |
| `expected` | `str` | Expected type name. |
| `got` | `str` | Actual type name. |

**Attributes:** `key`, `expected`, `got`

---

### NoCapableAgentError

```python
class NoCapableAgentError(SKTKError)
```

Raised when no agent can handle a given task.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `task_type` | `str` | The unhandled task type. |
| `available` | `list[str]` | Names of available agents. |

**Attributes:** `task_type`, `available`

---

### ContractValidationError

```python
class ContractValidationError(SKTKError)
```

Raised when LLM output fails to validate against an output contract.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `model_name` | `str` | Target Pydantic model name. |
| `raw_output` | `str` | Raw LLM text that failed validation. |
| `validation_errors` | `list[dict[str, Any]]` | Pydantic validation error details. |

**Attributes:** `model_name`, `raw_output`, `validation_errors`

---

### RetryExhaustedError

```python
class RetryExhaustedError(SKTKError)
```

Raised when all retry attempts have been exhausted.

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `attempts` | `int` | Number of attempts made. |
| `last_error` | `Exception` | The final exception. |

**Attributes:** `attempts`, `last_error`

---

## Resilience

Defined in `sktk.core.resilience`. Retry and circuit breaker patterns for production-grade reliability.

### BackoffStrategy

```python
class BackoffStrategy(Enum)
```

Backoff strategies for retries.

**Values**

| Value | Description |
|---|---|
| `FIXED` | Fixed delay between retries. |
| `EXPONENTIAL` | Exponential backoff (`base_delay * 2^attempt`). |
| `EXPONENTIAL_JITTER` | Exponential backoff with random jitter. |

---

### RetryPolicy

Configurable retry policy with backoff.

```python
@dataclass(frozen=True)
class RetryPolicy
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `3` | Maximum number of retry attempts. |
| `base_delay` | `float` | `1.0` | Base delay in seconds. |
| `max_delay` | `float` | `60.0` | Maximum delay cap in seconds. |
| `backoff` | `BackoffStrategy` | `EXPONENTIAL_JITTER` | Backoff strategy to use. |
| `retryable_exceptions` | `tuple[type[Exception], ...]` | `(Exception,)` | Exception types eligible for retry. |

**Methods**

```python
async def execute(
    self,
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any
) -> T
```

Execute `fn` with retries according to this policy.

| Parameter | Type | Description |
|---|---|---|
| `fn` | `Callable[..., Awaitable[T]]` | Async function to execute. |
| `*args` | `Any` | Positional arguments forwarded to `fn`. |
| `**kwargs` | `Any` | Keyword arguments forwarded to `fn`. |

**Returns:** `T` -- the return value of `fn`.

**Raises:** `RetryExhaustedError` -- if all retries are exhausted.

---

### CircuitState

```python
class CircuitState(Enum)
```

States of a circuit breaker.

**Values**

| Value | Description |
|---|---|
| `CLOSED` | Normal operation; failures are counted. |
| `OPEN` | All calls rejected immediately. |
| `HALF_OPEN` | One test call allowed to check recovery. |

---

### CircuitBreaker

Circuit breaker to avoid hammering a failing service.

```python
@dataclass
class CircuitBreaker
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `failure_threshold` | `int` | `5` | Number of failures before opening the circuit. |
| `recovery_timeout` | `float` | `30.0` | Seconds to wait before transitioning from OPEN to HALF_OPEN. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `state` | `CircuitState` | Current circuit state. Automatically transitions from OPEN to HALF_OPEN after `recovery_timeout`. |

**Methods**

```python
def record_success(self) -> None
```

Record a successful call. Resets failure count and closes the circuit.

```python
def record_failure(self) -> None
```

Record a failed call. Increments failure count; opens the circuit if threshold is reached.

```python
async def execute(
    self,
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any
) -> T
```

Execute `fn` through the circuit breaker. Records success or failure automatically.

| Parameter | Type | Description |
|---|---|---|
| `fn` | `Callable[..., Awaitable[T]]` | Async function to execute. |
| `*args` | `Any` | Positional arguments forwarded to `fn`. |
| `**kwargs` | `Any` | Keyword arguments forwarded to `fn`. |

**Returns:** `T` -- the return value of `fn`.

**Raises:** `RuntimeError` -- if the circuit is OPEN.

---

## Executor

Defined in `sktk.core.executor`. Thread pool executor for CPU-bound work. Wraps `asyncio.to_thread` and provides a managed thread pool for running blocking operations without stalling the event loop.

### run_in_executor

Run a sync function in a thread pool without blocking the event loop.

```python
async def run_in_executor(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> T
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `fn` | `Callable[..., T]` | Synchronous function to run. |
| `*args` | `Any` | Positional arguments forwarded to `fn`. |
| `**kwargs` | `Any` | Keyword arguments forwarded to `fn`. |

**Returns:** `T` -- the return value of `fn`.

---

### run_parallel

Run multiple sync functions in parallel using the thread pool.

```python
async def run_parallel(*tasks: Callable[..., T]) -> list[T]
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `*tasks` | `Callable[..., T]` | Zero-argument callables to execute concurrently. |

**Returns:** `list[T]` -- results in the same order as the input tasks.

---

## Secrets

Defined in `sktk.core.secrets`. Pluggable secrets management abstracting API key and credential access.

### SecretsProvider

Runtime-checkable protocol for secrets providers.

```python
@runtime_checkable
class SecretsProvider(Protocol)
```

**Methods**

```python
def get(self, key: str) -> str | None
```

Retrieve a secret by key. Returns `None` if not found.

```python
def require(self, key: str) -> str
```

Retrieve a secret by key. Raises `KeyError` if not found.

---

### EnvSecretsProvider

Load secrets from environment variables. Implements `SecretsProvider`.

```python
class EnvSecretsProvider
```

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prefix` | `str` | `""` | Prefix prepended to all key lookups (e.g. `"SKTK_"`). |

**Methods**

```python
def get(self, key: str) -> str | None
```

Look up `{prefix}{key}` in environment variables. Returns `None` if not set.

```python
def require(self, key: str) -> str
```

Look up `{prefix}{key}` in environment variables. Raises `KeyError` if not set.

---

### FileSecretsProvider

Load secrets from a dotenv-style file. Implements `SecretsProvider`.

```python
class FileSecretsProvider
```

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the dotenv file. Lines with `KEY=VALUE` are parsed; comments (`#`) and blank lines are skipped. Surrounding quotes on values are stripped. |

**Methods**

```python
def get(self, key: str) -> str | None
```

Look up a key in the parsed secrets. Returns `None` if not found.

```python
def require(self, key: str) -> str
```

Look up a key in the parsed secrets. Raises `KeyError` if not found.

---

### ChainedSecretsProvider

Try multiple providers in order, returning the first match. Implements `SecretsProvider`.

```python
class ChainedSecretsProvider
```

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `providers` | `list[SecretsProvider]` | Ordered list of providers to query. |

**Methods**

```python
def get(self, key: str) -> str | None
```

Query each provider in order. Returns the first non-`None` result, or `None` if all return `None`.

```python
def require(self, key: str) -> str
```

Query each provider in order. Raises `KeyError` (listing all provider class names) if no provider has the key.

---

## Config

Defined in `sktk.core.config`. Configuration management following 12-factor app principles.

### SKTKConfig

Top-level SKTK configuration. Loads settings from environment variables, YAML files, or dicts.

```python
@dataclass
class SKTKConfig
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `ModelConfig` | `ModelConfig()` | LLM model settings. |
| `retry` | `RetryConfig` | `RetryConfig()` | Retry policy settings. |
| `logging` | `LoggingConfig` | `LoggingConfig()` | Logging settings. |
| `default_timeout` | `float` | `60.0` | Default timeout in seconds. |
| `max_iterations` | `int` | `10` | Maximum agent loop iterations. |

**Class Methods**

```python
@classmethod
def from_env(cls, prefix: str = "SKTK_") -> SKTKConfig
```

Load configuration from environment variables. Variables follow the pattern `{prefix}SECTION_KEY` (e.g. `SKTK_MODEL_PROVIDER`, `SKTK_RETRY_MAX_RETRIES`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prefix` | `str` | `"SKTK_"` | Environment variable prefix. |

**Returns:** `SKTKConfig`

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> SKTKConfig
```

Load configuration from a dictionary. Keys: `"model"`, `"retry"`, `"logging"`, `"default_timeout"`, `"max_iterations"`.

| Parameter | Type | Description |
|---|---|---|
| `data` | `dict[str, Any]` | Configuration dictionary. |

**Returns:** `SKTKConfig`

```python
@classmethod
def from_yaml(cls, path: str | Path) -> SKTKConfig
```

Load configuration from a YAML file. Requires `PyYAML` (`pip install pyyaml`).

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the YAML file. |

**Returns:** `SKTKConfig`

**Raises:** `ImportError` -- if `PyYAML` is not installed.

---

### ModelConfig

LLM model configuration.

```python
@dataclass
class ModelConfig
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"openai"` | LLM provider name. |
| `model_name` | `str` | `"gpt-4"` | Model identifier. |
| `temperature` | `float` | `0.7` | Sampling temperature. |
| `max_tokens` | `int` | `4096` | Maximum tokens per request. |
| `api_key` | `str` | `""` | API key for the provider. |

---

### RetryConfig

Retry policy configuration.

```python
@dataclass
class RetryConfig
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `3` | Maximum number of retries. |
| `base_delay` | `float` | `1.0` | Base delay in seconds. |
| `max_delay` | `float` | `60.0` | Maximum delay cap in seconds. |
| `backoff` | `str` | `"exponential_jitter"` | Backoff strategy name. |

---

### LoggingConfig

Logging configuration.

```python
@dataclass
class LoggingConfig
```

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | `"INFO"` | Log level. |
| `structured` | `bool` | `True` | Whether to use structured (JSON) logging. |
