# sktk.agent

Core agent abstraction, tools, filters, contracts, and supporting infrastructure for building LLM-powered agents on top of Semantic Kernel.

## Table of Contents

- [SKTKAgent](#sktk agent)
- [Tool](#tool)
- [tool (decorator)](#tool-decorator)
- [AgentFilter (Protocol)](#agentfilter-protocol)
- [FilterContext](#filtercontext)
- [ContentSafetyFilter](#contentsafetyfilter)
- [PIIFilter](#piifilter)
- [TokenBudgetFilter](#tokenbudgetfilter)
- [PromptInjectionFilter](#promptinjectionfilter)
- [run_filter_pipeline](#run_filter_pipeline)
- [serialize_input](#serialize_input)
- [parse_output](#parse_output)
- [LifecycleHooks](#lifecyclehooks)
- [MiddlewareStack](#middlewarestack)
- [timing_middleware](#timing_middleware)
- [logging_middleware](#logging_middleware)
- [Capability](#capability)
- [match_capabilities](#match_capabilities)
- [LLMProvider (Protocol)](#llmprovider-protocol)
- [ProviderRegistry](#providerregistry)
- [register_provider](#register_provider)
- [create_provider](#create_provider)
- [get_registry](#get_registry)
- [PermissionPolicy](#permissionpolicy)
- [RateLimitPolicy](#ratelimitpolicy)
- [StepStatus (Enum)](#stepstatus-enum)
- [PlanStep](#planstep)
- [Plan](#plan)
- [TaskPlanner](#taskplanner)
- [PromptTemplate](#prompttemplate)
- [load_prompt](#load_prompt)
- [load_prompts](#load_prompts)
- [load_agent_from_dict](#load_agent_from_dict)
- [load_agent_from_yaml](#load_agent_from_yaml)
- [load_agent_from_json](#load_agent_from_json)
- [register_filter](#register_filter)
- [ApprovalGate](#approvalgate)
- [ApprovalRequest](#approvalrequest)
- [AutoApprovalFilter](#autoapprovalfilter)
- [FallbackChain](#fallbackchain)
- [tools_from_openapi](#tools_from_openapi)
- [tools_from_openapi_file](#tools_from_openapi_file)

---

## Agent

### SKTKAgent

High-level agent wrapping Semantic Kernel's `ChatCompletionAgent`.

```python
@dataclass
class SKTKAgent
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Agent identifier. |
| `instructions` | `str` | System instructions / persona for the agent. |
| `session` | `Session \| None` | Optional session for conversation history tracking. Default `None`. |
| `capabilities` | `list[Capability]` | Declared capabilities of this agent. Default `[]`. |
| `input_contract` | `type[BaseModel] \| None` | Pydantic model for typed input validation. Default `None`. |
| `output_contract` | `type[BaseModel] \| None` | Pydantic model for typed output parsing. Default `None`. |
| `filters` | `list[AgentFilter]` | Guardrail filter pipeline applied to input and output. Default `[]`. |
| `tools` | `list[Tool]` | Tools the agent can invoke via function calling. Default `[]`. |
| `hooks` | `LifecycleHooks` | Lifecycle event hooks. Default `LifecycleHooks()`. |
| `max_iterations` | `int` | Maximum agent loop iterations. Default `10`. |
| `timeout` | `float` | Invocation timeout in seconds. Default `60.0`. |

#### Methods

##### `with_responses`

```python
@classmethod
def with_responses(cls, name: str, responses: list[str], **kwargs: Any) -> SKTKAgent
```

Create a test agent with scripted responses (no LLM needed). Useful for deterministic testing.

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Agent name. |
| `responses` | `list[str]` | Scripted responses returned in order. |
| `**kwargs` | `Any` | Additional `SKTKAgent` constructor arguments. |

**Returns:** `SKTKAgent`

---

##### `get_tool`

```python
def get_tool(self, name: str) -> Tool | None
```

Look up a registered tool by name.

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Tool name to look up. |

**Returns:** `Tool | None`

---

##### `call_tool`

```python
async def call_tool(self, name: str, **kwargs: Any) -> Any
```

Call a registered tool by name. Raises `KeyError` if the tool is not registered.

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Tool name to invoke. |
| `**kwargs` | `Any` | Arguments passed to the tool function. |

**Returns:** `Any`

---

##### `invoke`

```python
async def invoke(self, message: str | BaseModel, **kwargs: Any) -> Any
```

Invoke the agent with a string or typed input. Runs the full pipeline: input filters, LLM call, output filters, output contract parsing, and lifecycle hooks.

| Param | Type | Description |
|---|---|---|
| `message` | `str \| BaseModel` | User message or Pydantic model (serialized via `serialize_input`). |
| `**kwargs` | `Any` | Additional arguments passed to the underlying LLM call. |

**Returns:** `Any` -- raw string or parsed `BaseModel` if `output_contract` is set.

---

##### `invoke_stream`

```python
async def invoke_stream(self, message: str | BaseModel, **kwargs: Any) -> AsyncIterator[str]
```

Invoke the agent and yield response chunks as they arrive.

| Param | Type | Description |
|---|---|---|
| `message` | `str \| BaseModel` | User message or Pydantic model. |
| `**kwargs` | `Any` | Additional arguments passed to the underlying LLM call. |

**Returns:** `AsyncIterator[str]`

---

##### `__rshift__`

```python
def __rshift__(self, other: Any) -> Any
```

Support `>>` operator for pipeline topology DSL. Connects agents sequentially or in parallel.

| Param | Type | Description |
|---|---|---|
| `other` | `SKTKAgent \| list \| AgentNode \| SequentialNode \| ParallelNode` | Next node(s) in the pipeline. |

**Returns:** `SequentialNode`

---

##### Async Context Manager

`SKTKAgent` supports `async with` for resource cleanup:

```python
async with SKTKAgent(name="a", instructions="...") as agent:
    result = await agent.invoke("Hello")
```

---

## Tools

### Tool

A callable tool that an agent can invoke via function calling.

```python
@dataclass(frozen=True)
class Tool
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Tool identifier. |
| `description` | `str` | Human-readable description (sent to LLM). |
| `fn` | `Callable[..., Any]` | The underlying function. |
| `parameters` | `dict[str, Any]` | JSON-schema-like parameter definition. Default `{}`. |

#### Methods

##### `__call__`

```python
async def __call__(self, **kwargs: Any) -> Any
```

Invoke the underlying function. Awaits the result if the function is async.

---

##### `to_schema`

```python
def to_schema(self) -> dict[str, Any]
```

Export as a JSON-schema-like dict for LLM function-calling.

**Returns:** `dict[str, Any]` with keys `name`, `description`, `parameters`.

---

### tool (decorator)

```python
def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]
```

Decorator to create a `Tool` from a function. If `name` is not provided, uses the function name. If `parameters` is not provided, infers them from the function signature.

| Param | Type | Description |
|---|---|---|
| `name` | `str \| None` | Override tool name. Default: function `__name__`. |
| `description` | `str` | Tool description. Default: function docstring or `""`. |
| `parameters` | `dict[str, Any] \| None` | Explicit JSON-schema parameters. Default: auto-inferred. |

**Returns:** `Callable[[Callable[..., Any]], Tool]`

---

### tools_from_openapi

```python
def tools_from_openapi(spec: dict[str, Any]) -> list[Tool]
```

Generate `Tool` objects from an OpenAPI specification dict. Each operation becomes a tool with a stub function.

| Param | Type | Description |
|---|---|---|
| `spec` | `dict[str, Any]` | Parsed OpenAPI specification. |

**Returns:** `list[Tool]`

---

### tools_from_openapi_file

```python
def tools_from_openapi_file(path: str | Path) -> list[Tool]
```

Load tools from an OpenAPI JSON or YAML file. YAML support requires PyYAML.

| Param | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the OpenAPI spec file. |

**Returns:** `list[Tool]`

---

## Filters

### AgentFilter (Protocol)

```python
@runtime_checkable
class AgentFilter(Protocol)
```

Protocol for guardrail filters. Implement all three methods to create a custom filter.

#### Methods

```python
async def on_input(self, context: FilterContext) -> FilterResult
async def on_function_call(self, context: FilterContext) -> FilterResult
async def on_output(self, context: FilterContext) -> FilterResult
```

Each method receives a `FilterContext` and returns one of `Allow`, `Deny`, or `Modify`.

---

### FilterContext

Context passed through the filter pipeline.

```python
@dataclass
class FilterContext
```

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The content being filtered. |
| `stage` | `FilterStage` | One of `"input"`, `"output"`, `"function_call"`. |
| `agent_name` | `str \| None` | Name of the agent. Default `None`. |
| `token_count` | `int \| None` | Token count (if available). Default `None`. |
| `metadata` | `dict` | Arbitrary metadata. Default `{}`. |

`FilterStage` is defined as:

```python
FilterStage = Literal["input", "output", "function_call"]
```

---

### ContentSafetyFilter

Block content matching configurable regex patterns.

```python
class ContentSafetyFilter
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `blocked_patterns` | `list[str]` | Regex patterns to block. |

#### Methods

```python
async def on_input(self, context: FilterContext) -> FilterResult
async def on_function_call(self, context: FilterContext) -> FilterResult
async def on_output(self, context: FilterContext) -> FilterResult
```

Returns `Deny` if content matches any blocked pattern, `Allow` otherwise.

---

### PIIFilter

Detect common PII patterns (email, phone, SSN).

```python
class PIIFilter
```

#### Constructor Parameters

None.

#### Methods

```python
async def on_input(self, context: FilterContext) -> FilterResult
async def on_function_call(self, context: FilterContext) -> FilterResult
async def on_output(self, context: FilterContext) -> FilterResult
```

Returns `Deny` if content contains any PII pattern, `Allow` otherwise.

---

### TokenBudgetFilter

Deny requests exceeding a token budget.

```python
class TokenBudgetFilter
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `max_tokens` | `int` | Maximum allowed token count. |

#### Methods

```python
async def on_input(self, context: FilterContext) -> FilterResult
async def on_function_call(self, context: FilterContext) -> FilterResult
async def on_output(self, context: FilterContext) -> FilterResult
```

Checks `context.token_count` against the budget on input. Function call and output stages always return `Allow`.

---

### PromptInjectionFilter

Detect common prompt injection and jailbreak patterns. Catches attempts to override system instructions, extract system prompts, or bypass guardrails.

```python
class PromptInjectionFilter
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `extra_patterns` | `list[str] \| None` | Additional regex patterns to detect. Default `None`. |

#### Methods

```python
async def on_input(self, context: FilterContext) -> FilterResult
async def on_function_call(self, context: FilterContext) -> FilterResult
async def on_output(self, context: FilterContext) -> FilterResult
```

Returns `Deny` on input and function_call stages if injection is detected. Output stage always returns `Allow`.

---

### run_filter_pipeline

```python
async def run_filter_pipeline(
    filters: list[AgentFilter],
    context: FilterContext,
) -> FilterResult
```

Execute filters in order. `Deny` short-circuits the pipeline. `Modify` updates the content for subsequent filters.

| Param | Type | Description |
|---|---|---|
| `filters` | `list[AgentFilter]` | Ordered list of filters. |
| `context` | `FilterContext` | Initial filter context. |

**Returns:** `FilterResult` (`Allow`, `Deny`, or `Modify`)

---

## Contracts

### serialize_input

```python
def serialize_input(model: BaseModel, template: str | None = None) -> str
```

Convert a Pydantic model to a prompt string. If `template` is provided, uses `str.format()` with the model fields. Otherwise, renders as `**key**: value` lines.

| Param | Type | Description |
|---|---|---|
| `model` | `BaseModel` | Pydantic model to serialize. |
| `template` | `str \| None` | Optional format string. Default `None`. |

**Returns:** `str`

---

### parse_output

```python
def parse_output(raw: str, model: type[T]) -> T
```

Parse LLM output into a validated Pydantic model. Extracts JSON from raw text (supports bare JSON, fenced code blocks, and embedded JSON objects). Raises `ContractValidationError` on failure.

| Param | Type | Description |
|---|---|---|
| `raw` | `str` | Raw LLM output text. |
| `model` | `type[T]` | Pydantic model class (where `T` is bound to `BaseModel`). |

**Returns:** `T`

---

## Hooks / Middleware

### LifecycleHooks

Collection of lifecycle hooks for an agent. Hooks run at key points: before invocation, after success, and after errors.

```python
@dataclass
class LifecycleHooks
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `on_start` | `list[OnStartHook]` | Hooks called before invocation. Default `[]`. |
| `on_complete` | `list[OnCompleteHook]` | Hooks called after successful invocation. Default `[]`. |
| `on_error` | `list[OnErrorHook]` | Hooks called after failed invocation. Default `[]`. |

Hook type signatures:

```python
OnStartHook = Callable[[str, str], Awaitable[None]]           # (agent_name, input)
OnCompleteHook = Callable[[str, str, Any], Awaitable[None]]   # (agent_name, input, output)
OnErrorHook = Callable[[str, str, Exception], Awaitable[None]] # (agent_name, input, error)
```

#### Methods

##### `fire_start`

```python
async def fire_start(self, agent_name: str, input_text: str) -> None
```

Invoke all `on_start` hooks before agent invocation.

---

##### `fire_complete`

```python
async def fire_complete(self, agent_name: str, input_text: str, output: Any) -> None
```

Invoke all `on_complete` hooks after successful invocation.

---

##### `fire_error`

```python
async def fire_error(self, agent_name: str, input_text: str, error: Exception) -> None
```

Invoke all `on_error` hooks after a failed invocation.

---

### MiddlewareStack

Composable middleware stack for agent invocations. Wraps agent calls to add cross-cutting concerns (timing, caching, logging, error handling) without modifying agents.

```python
class MiddlewareStack
```

Middleware function signature:

```python
Middleware = Callable[[str, str, InvokeNext], Awaitable[Any]]
InvokeNext = Callable[[str], Awaitable[Any]]
```

#### Methods

##### `add`

```python
def add(self, mw: Middleware) -> None
```

Add middleware to the stack.

---

##### `use`

```python
def use(self, mw: Middleware) -> Middleware
```

Decorator to register middleware. Returns the middleware unchanged.

---

##### `wrap`

```python
def wrap(
    self, invoke_fn: Callable[..., Awaitable[Any]], agent_name: str = ""
) -> Callable[..., Awaitable[Any]]
```

Wrap an invoke function with the middleware stack. First-added middleware runs outermost.

| Param | Type | Description |
|---|---|---|
| `invoke_fn` | `Callable[..., Awaitable[Any]]` | The function to wrap (e.g. `agent.invoke`). |
| `agent_name` | `str` | Agent name passed to middleware. Default `""`. |

**Returns:** `Callable[..., Awaitable[Any]]`

---

### timing_middleware

```python
async def timing_middleware(agent_name: str, message: str, next_fn: InvokeNext) -> Any
```

Built-in middleware that tracks invocation duration.

---

### logging_middleware

```python
async def logging_middleware(agent_name: str, message: str, next_fn: InvokeNext) -> Any
```

Built-in middleware that logs invocations via structured logger (`sktk.observability.logging`).

---

## Capabilities

### Capability

Structured declaration of what an agent can do.

```python
@dataclass
class Capability
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Capability identifier. |
| `description` | `str` | Human-readable description. |
| `input_types` | `list[type[BaseModel]]` | Accepted input model types. |
| `output_types` | `list[type[BaseModel]]` | Produced output model types. |
| `tags` | `list[str]` | Searchable tags. Default `[]`. |

---

### match_capabilities

```python
def match_capabilities(
    capabilities: list[Capability],
    *,
    input_type: type[BaseModel] | None = None,
    tags: list[str] | None = None,
) -> list[Capability]
```

Find capabilities matching input type and/or tags.

| Param | Type | Description |
|---|---|---|
| `capabilities` | `list[Capability]` | Capabilities to search. |
| `input_type` | `type[BaseModel] \| None` | Filter by accepted input type. |
| `tags` | `list[str] \| None` | Filter by tag overlap (OR match). |

**Returns:** `list[Capability]`

---

## Providers

### LLMProvider (Protocol)

```python
@runtime_checkable
class LLMProvider(Protocol)
```

Protocol for LLM provider backends. Implement this to integrate any LLM service (OpenAI, Anthropic, Azure, local models, etc.).

#### Properties

##### `name`

```python
@property
def name(self) -> str
```

Provider identifier (e.g. `"openai"`, `"anthropic"`).

#### Methods

##### `complete`

```python
async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str
```

Send messages and return the completion text.

---

### ProviderRegistry

Registry and factory for LLM providers.

```python
class ProviderRegistry
```

#### Methods

##### `register`

```python
def register(self, name: str, factory: type[LLMProvider]) -> None
```

Register a provider class under a name.

---

##### `create`

```python
def create(self, name: str, **kwargs: Any) -> LLMProvider
```

Create a provider instance by registered name. Raises `KeyError` if name is unknown.

---

##### `available`

```python
@property
def available(self) -> list[str]
```

List registered provider names.

---

### register_provider

```python
def register_provider(name: str, factory: type[LLMProvider]) -> None
```

Register a provider in the default global registry.

---

### create_provider

```python
def create_provider(name: str, **kwargs: Any) -> LLMProvider
```

Create a provider from the default global registry.

---

### get_registry

```python
def get_registry() -> ProviderRegistry
```

Get the default provider registry.

---

## Permissions

### PermissionPolicy

Policy defining which functions an agent can call. Supports allowlist, denylist, or both (allowlist takes precedence). Implements the `AgentFilter` protocol.

```python
class PermissionPolicy
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `allow` | `list[str] \| None` | Allowlist of permitted function names. Default `None` (no allowlist). |
| `deny` | `list[str] \| None` | Denylist of forbidden function names. Default `None`. |
| `audit_trail` | `AuditTrail \| None` | Optional audit trail used to record denied calls as `permission_denied` events. |

#### Methods

##### `on_input`

```python
async def on_input(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_output`

```python
async def on_output(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_function_call`

```python
async def on_function_call(self, context: FilterContext) -> FilterResult
```

Check if the function call is permitted. Reads `function_name` from `context.metadata`.

If an `AuditTrail` is configured, denied calls emit `permission_denied` entries so you can trace which plugin invocations were blocked.

---

##### `is_allowed`

```python
def is_allowed(self, function_name: str) -> bool
```

Quick synchronous check if a function is permitted.

---

### RateLimitPolicy

Rate limit on agent invocations per time window. Tracks call counts and denies when limit is exceeded. Implements the `AgentFilter` protocol.

```python
@dataclass
class RateLimitPolicy
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `max_calls` | `int` | Maximum calls allowed per window. Default `100`. |
| `window_seconds` | `float` | Time window in seconds. Default `60.0`. |
| `audit_trail` | `AuditTrail \| None` | Optional audit trail that logs rate limit denials as `rate_limit_exceeded`. |

#### Methods

##### `on_input`

```python
async def on_input(self, context: FilterContext) -> FilterResult
```

Enforces rate limiting. Returns `Deny` if the limit is exceeded.

If an `AuditTrail` is provided, those denials emit `rate_limit_exceeded` entries so you can track throttling decisions.

---

##### `on_function_call`

```python
async def on_function_call(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_output`

```python
async def on_output(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

## Planner

### StepStatus (Enum)

```python
class StepStatus(Enum)
```

| Value | String |
|---|---|
| `PENDING` | `"pending"` |
| `IN_PROGRESS` | `"in_progress"` |
| `COMPLETED` | `"completed"` |
| `FAILED` | `"failed"` |
| `SKIPPED` | `"skipped"` |

---

### PlanStep

A single step in an execution plan.

```python
@dataclass
class PlanStep
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique step identifier. |
| `description` | `str` | Human-readable description. |
| `tool_name` | `str \| None` | Tool to invoke. Default `None`. |
| `tool_args` | `dict[str, Any]` | Arguments for the tool. Default `{}`. |
| `depends_on` | `list[str]` | IDs of steps that must complete first. Default `[]`. |
| `status` | `StepStatus` | Current status. Default `StepStatus.PENDING`. |
| `result` | `Any` | Execution result. Default `None`. |
| `error` | `str \| None` | Error message if failed. Default `None`. |

#### Properties

##### `is_ready`

```python
@property
def is_ready(self) -> bool
```

Check if all dependencies are satisfied (status is `PENDING`).

---

### Plan

An ordered execution plan with dependency tracking.

```python
@dataclass
class Plan
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `goal` | `str` | The high-level goal this plan achieves. |
| `steps` | `list[PlanStep]` | Ordered list of plan steps. Default `[]`. |

#### Methods

##### `add_step`

```python
def add_step(self, step: PlanStep) -> None
```

Append a step to the plan.

---

##### `get_step`

```python
def get_step(self, step_id: str) -> PlanStep | None
```

Look up a step by ID.

---

##### `ready_steps`

```python
def ready_steps(self) -> list[PlanStep]
```

Return steps whose dependencies are all completed.

---

##### `complete_step`

```python
def complete_step(self, step_id: str, result: Any = None) -> None
```

Mark a step as completed. Raises `KeyError` if step not found.

---

##### `fail_step`

```python
def fail_step(self, step_id: str, error: str) -> None
```

Mark a step as failed. Raises `KeyError` if step not found.

---

##### `is_complete`

```python
@property
def is_complete(self) -> bool
```

`True` if all steps are completed or skipped.

---

##### `progress`

```python
@property
def progress(self) -> float
```

Fraction of steps completed or skipped (0.0 to 1.0).

---

##### `to_dict`

```python
def to_dict(self) -> dict[str, Any]
```

Serialize the plan to a dictionary with `goal`, `steps`, and `progress`.

---

### TaskPlanner

Creates execution plans from a goal. Provides manual plan construction and execution tracking.

```python
class TaskPlanner
```

#### Methods

##### `create_plan`

```python
def create_plan(self, goal: str, steps: list[dict[str, Any]] | None = None) -> Plan
```

Create a plan from explicit steps.

| Param | Type | Description |
|---|---|---|
| `goal` | `str` | High-level goal description. |
| `steps` | `list[dict[str, Any]] \| None` | Step definitions. Each dict may have `id`, `description`, `tool_name`, `tool_args`, `depends_on`. |

**Returns:** `Plan`

---

##### `execute_plan`

```python
async def execute_plan(self, plan: Plan, executor: Any) -> Plan
```

Execute a plan step by step using the provided executor. The executor should be callable with `(step: PlanStep) -> Any`.

| Param | Type | Description |
|---|---|---|
| `plan` | `Plan` | The plan to execute. |
| `executor` | `Any` | Async callable that executes a `PlanStep`. |

**Returns:** `Plan`

---

## Templates

### PromptTemplate

A reusable prompt template with `{{ var }}` variable substitution.

```python
@dataclass(frozen=True)
class PromptTemplate
```

#### Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Template identifier. |
| `text` | `str` | Template text with `{{ var }}` placeholders. |
| `defaults` | `dict[str, str]` | Default variable values. Default `{}`. |
| `metadata` | `dict[str, Any]` | Arbitrary metadata. Default `{}`. |

#### Properties

##### `variables`

```python
@property
def variables(self) -> list[str]
```

Extract all variable names from the template.

#### Methods

##### `render`

```python
def render(self, **kwargs: str) -> str
```

Render the template with given variables. Merges with defaults. Raises `ValueError` if required variables are missing.

---

##### `validate`

```python
def validate(self) -> list[str]
```

Validate template, returning a list of issues (empty if valid). Checks for unmatched braces and invalid variable names.

---

### load_prompt

```python
def load_prompt(path: str | Path) -> PromptTemplate
```

Load a prompt template from a `.prompt` file. Supports optional YAML frontmatter for `name`, `defaults`, and metadata.

| Param | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.prompt` file. |

**Returns:** `PromptTemplate`

File format:

```
---
name: analyze
defaults:
  format: markdown
---
Analyze the following {{topic}} data: {{data}}
Output in {{format}} format.
```

---

### load_prompts

```python
def load_prompts(directory: str | Path) -> dict[str, PromptTemplate]
```

Load all `.prompt` files from a directory.

| Param | Type | Description |
|---|---|---|
| `directory` | `str \| Path` | Directory containing `.prompt` files. |

**Returns:** `dict[str, PromptTemplate]` keyed by template name.

---

## Loader

### load_agent_from_dict

```python
def load_agent_from_dict(config: dict[str, Any]) -> SKTKAgent
```

Create an `SKTKAgent` from a configuration dictionary.

| Param | Type | Description |
|---|---|---|
| `config` | `dict[str, Any]` | Agent configuration. Required keys: `name`, `instructions`. Optional: `max_iterations`, `timeout`, `filters`, `capabilities`, `session_id`. |

**Returns:** `SKTKAgent`

Expected config structure:

```python
{
    "name": "analyst",
    "instructions": "You analyze data.",
    "max_iterations": 5,
    "timeout": 30.0,
    "filters": [
        {"type": "prompt_injection"},
        {"type": "pii"},
        {"type": "content_safety", "blocked_patterns": ["badword"]},
        {"type": "token_budget", "max_tokens": 4000},
    ],
    "capabilities": [
        {"name": "analysis", "description": "Analyze data", "tags": ["finance"]},
    ],
}
```

Built-in filter types: `content_safety`, `pii`, `prompt_injection`, `token_budget`.

---

### load_agent_from_yaml

```python
def load_agent_from_yaml(path: str | Path) -> SKTKAgent
```

Load an `SKTKAgent` from a YAML file. Requires PyYAML.

| Param | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to YAML config file. |

**Returns:** `SKTKAgent`

---

### load_agent_from_json

```python
def load_agent_from_json(path: str | Path) -> SKTKAgent
```

Load an `SKTKAgent` from a JSON file.

| Param | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to JSON config file. |

**Returns:** `SKTKAgent`

---

### register_filter

```python
def register_filter(name: str, cls: type) -> None
```

Register a custom filter class for use in declarative agent definitions (via `load_agent_from_dict`).

| Param | Type | Description |
|---|---|---|
| `name` | `str` | Filter type name used in config dicts. |
| `cls` | `type` | Filter class to instantiate. |

---

## Approval

### ApprovalRequest

A pending approval request.

```python
@dataclass
class ApprovalRequest
```

| Field | Type | Description |
|---|---|---|
| `agent_name` | `str` | Agent requesting approval. |
| `action` | `str` | Action requiring approval. |
| `details` | `dict[str, Any]` | Additional context. Default `{}`. |
| `approved` | `bool \| None` | Approval result. Default `None` (pending). |

---

### ApprovalGate

Gate that pauses execution until human approval. Implements the `AgentFilter` protocol.

```python
class ApprovalGate
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `timeout` | `float` | Seconds to wait before auto-denying. Default `300.0`. |
| `audit_trail` | `AuditTrail \| None` | Optional audit trail used to log approval requests, decisions, and timeouts. |

#### Properties

##### `pending`

```python
@property
def pending(self) -> ApprovalRequest | None
```

The currently pending approval request, or `None`.

#### Methods

##### `approve`

```python
def approve(self) -> None
```

Approve the pending request.

---

##### `deny`

```python
def deny(self, reason: str = "Denied by human reviewer") -> None
```

Deny the pending request.

---

##### `reset`

```python
def reset(self) -> None
```

Reset the gate for reuse.

---

##### `wait_for_approval`

```python
async def wait_for_approval(
    self, agent_name: str, action: str, details: dict[str, Any] | None = None
) -> bool
```

Wait for human approval. Returns `True` if approved, `False` on denial or timeout.

When an audit trail is configured, the gate emits `approval_requested`, `approval_granted`, `approval_denied`, and `approval_timeout` events so all approvals are traceable.

---

##### `on_input`

```python
async def on_input(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_output`

```python
async def on_output(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_function_call`

```python
async def on_function_call(self, context: FilterContext) -> FilterResult
```

Gate function calls pending human approval. Reads `function_name` from `context.metadata`.

---

### AutoApprovalFilter

Filter that auto-approves safe functions, gates dangerous ones through an `ApprovalGate`. Implements the `AgentFilter` protocol.

```python
class AutoApprovalFilter
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `safe_functions` | `list[str]` | Function names that are auto-approved. |
| `gate` | `ApprovalGate` | Gate used for non-safe functions. |

#### Methods

##### `on_input`

```python
async def on_input(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_output`

```python
async def on_output(self, context: FilterContext) -> FilterResult
```

Always returns `Allow`.

---

##### `on_function_call`

```python
async def on_function_call(self, context: FilterContext) -> FilterResult
```

Auto-approves safe functions. Delegates to the `ApprovalGate` for all others.

---

## FallbackChain

### FallbackChain

Try agents in sequence, falling back on failure. Common pattern for production resilience.

```python
class FallbackChain
```

#### Constructor Parameters

| Param | Type | Description |
|---|---|---|
| `agents` | `list[SKTKAgent]` | Ordered list of agents to try. Must be non-empty. |
| `fallback_exceptions` | `tuple[type[Exception], ...]` | Exception types that trigger fallback. Default `(Exception,)`. |

#### Properties

##### `agents`

```python
@property
def agents(self) -> list[SKTKAgent]
```

Copy of the agent list.

#### Methods

##### `invoke`

```python
async def invoke(self, message: str, **kwargs: Any) -> Any
```

Try each agent in order until one succeeds. Raises the last exception if all fail.

| Param | Type | Description |
|---|---|---|
| `message` | `str` | User message. |
| `**kwargs` | `Any` | Additional arguments passed to `agent.invoke`. |

**Returns:** `Any`
