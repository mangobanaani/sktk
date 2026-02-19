# sktk.observability

Tracing, metrics, events, audit trails, structured logging, quota enforcement, and profiling.

## Table of Contents

- [EventSink (Protocol)](#eventsink-protocol)
- [EventStream](#eventstream)
- [ContextLogger](#contextlogger)
- [StructuredFormatter](#structuredformatter)
- [get_logger](#get_logger)
- [configure_structured_logging](#configure_structured_logging)
- [PricingModel](#pricingmodel)
- [TokenTracker](#tokentracker)
- [AuditEntry](#auditentry)
- [AuditBackend (Protocol)](#auditbackend-protocol)
- [InMemoryAuditBackend](#inmemoryauditbackend)
- [AuditTrail](#audittrail)
- [ProfileEntry](#profileentry)
- [AgentProfiler](#agentprofiler)
- [ReplayEntry](#replayentry)
- [SessionRecorder](#sessionrecorder)
- [TokenQuota](#tokenquota)
- [TokenQuotaFilter](#tokenquotafilter)
- [instrument](#instrument)
- [create_span](#create_span)

---

## EventSink (Protocol)

```python
@runtime_checkable
class EventSink(Protocol)
```

Protocol for event sinks (stdout, log, message queue, etc.).

### Methods

#### `send`

```python
async def send(self, event: Any) -> None
```

Deliver a single event to the sink.

---

## EventStream

Collects typed events and optionally forwards them to registered sinks.

### Constructor

```python
def __init__(self, sinks: list[EventSink] | None = None) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `sinks` | `list[EventSink] \| None` | Optional list of sinks to forward events to. |

### Properties

#### `events`

```python
@property
def events(self) -> list[Any]
```

Return the list of collected events.

### Methods

#### `emit`

```python
async def emit(self, event: Any) -> None
```

Append an event and forward it to all registered sinks.

| Param | Type | Description |
|-------|------|-------------|
| `event` | `Any` | The event to emit. |

#### `clear`

```python
def clear(self) -> None
```

Discard all collected events.

#### `__iter__`

```python
def __iter__(self) -> Iterator[Any]
```

Iterate over collected events.

#### `__len__`

```python
def __len__(self) -> int
```

Return the number of collected events.

---

## ContextLogger

Logger that automatically enriches log records with fields from the current `ExecutionContext` (correlation_id, session_id, tenant_id, user_id).

### Constructor

```python
def __init__(self, name: str) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `name` | `str` | Logger name (passed to `logging.getLogger`). |

### Methods

#### `debug`

```python
def debug(self, msg: str, **kwargs: Any) -> None
```

Log at DEBUG level. Extra keyword arguments are attached as context fields.

#### `info`

```python
def info(self, msg: str, **kwargs: Any) -> None
```

Log at INFO level.

#### `warning`

```python
def warning(self, msg: str, **kwargs: Any) -> None
```

Log at WARNING level.

#### `error`

```python
def error(self, msg: str, **kwargs: Any) -> None
```

Log at ERROR level.

---

## StructuredFormatter

```python
class StructuredFormatter(logging.Formatter)
```

JSON log formatter that outputs structured log records including timestamp, level, logger name, message, context fields, and error info.

### Methods

#### `format`

```python
def format(self, record: logging.LogRecord) -> str
```

Serialize a log record to a JSON string with context fields.

**Returns:** `str` -- JSON-encoded log line.

---

## get_logger

```python
def get_logger(name: str) -> ContextLogger
```

Get a context-aware structured logger.

| Param | Type | Description |
|-------|------|-------------|
| `name` | `str` | Logger name. |

**Returns:** `ContextLogger`

---

## configure_structured_logging

```python
def configure_structured_logging(level: int = logging.INFO) -> None
```

Configure the root `sktk` logger with structured JSON output via `StructuredFormatter`.

| Param | Type | Description |
|-------|------|-------------|
| `level` | `int` | Logging level. Default `logging.INFO`. |

---

## PricingModel

Dataclass. Maps model names to per-token pricing (USD per 1K tokens).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `prices` | `dict[str, dict[str, float]]` | Mapping of model name to `{"prompt": float, "completion": float}`. Default empty dict. |

### Methods

#### `calculate`

```python
def calculate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None
```

Return the USD cost for the given token counts, or `None` if the model has no pricing entry.

| Param | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name to look up. |
| `prompt_tokens` | `int` | Number of prompt tokens. |
| `completion_tokens` | `int` | Number of completion tokens. |

**Returns:** `float | None`

---

## TokenTracker

Tracks token usage across agents and sessions, optionally enriching records with cost via a `PricingModel`.

### Constructor

```python
def __init__(self, pricing: PricingModel | None = None) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `pricing` | `PricingModel \| None` | Optional pricing model for cost attribution. |

### Methods

#### `record`

```python
def record(self, agent_name: str, session_id: str, model: str, usage: TokenUsage) -> None
```

Record a token usage entry.

| Param | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Name of the agent. |
| `session_id` | `str` | Session identifier. |
| `model` | `str` | Model name. |
| `usage` | `TokenUsage` | Token usage data. |

#### `get_usage`

```python
def get_usage(
    self,
    session_id: str,
    agent_name: str | None = None,
    model: str | None = None,
) -> TokenUsage
```

Return aggregated token usage, optionally filtered by agent or model.

| Param | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Session to aggregate for (required). |
| `agent_name` | `str \| None` | Optional agent name filter. |
| `model` | `str \| None` | Optional model name filter. |

**Returns:** `TokenUsage`

#### `clear`

```python
def clear(self, session_id: str | None = None) -> None
```

Remove records. If `session_id` is provided, only that session's records are removed; otherwise all records are cleared.

---

## AuditEntry

Frozen dataclass. A single audit log entry.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `str` | ISO-format timestamp. |
| `action` | `str` | Action name (e.g. `"invoke"`). |
| `agent_name` | `str` | Agent that performed the action. |
| `session_id` | `str` | Session identifier. |
| `user_id` | `str` | User identifier. |
| `correlation_id` | `str` | Correlation identifier. |
| `details` | `dict[str, Any]` | Additional action details. Default empty dict. |
| `outcome` | `str` | Outcome label. Default `"success"`. |
| `duration_ms` | `float` | Duration in milliseconds. Default `0.0`. |
| `previous_hash` | `str` | Hash of the preceding entry (chain integrity). Default `""`. |
| `entry_hash` | `str` | SHA-256 hash of this entry. Default `""`. |

### Methods

#### `to_dict`

```python
def to_dict(self) -> dict[str, Any]
```

Convert this audit entry to a plain dictionary.

---

## AuditBackend (Protocol)

```python
@runtime_checkable
class AuditBackend(Protocol)
```

Protocol for audit trail storage backends.

### Methods

#### `write`

```python
async def write(self, entry: AuditEntry) -> None
```

Persist a single audit entry.

#### `query`

```python
async def query(
    self,
    session_id: str | None = None,
    agent_name: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[AuditEntry]
```

Query entries with optional filters.

---

## InMemoryAuditBackend

In-memory audit backend for development and testing. Implements `AuditBackend`.

### Constructor

```python
def __init__(self) -> None
```

No parameters.

### Methods

#### `write`

```python
async def write(self, entry: AuditEntry) -> None
```

Append an audit entry to the in-memory store.

#### `query`

```python
async def query(
    self,
    session_id: str | None = None,
    agent_name: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[AuditEntry]
```

Return entries matching the given filters, capped at `limit`.

---

## AuditTrail

Records and queries agent actions with tamper-evident SHA-256 hash chaining.

### Constructor

```python
def __init__(self, backend: AuditBackend | None = None) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `backend` | `AuditBackend \| None` | Storage backend. Defaults to `InMemoryAuditBackend`. |

### Methods

#### `record`

```python
async def record(
    self,
    action: str,
    agent_name: str,
    session_id: str = "",
    user_id: str = "",
    correlation_id: str = "",
    details: dict[str, Any] | None = None,
    outcome: str = "success",
    duration_ms: float = 0.0,
) -> AuditEntry
```

Record an audit entry with tamper-evident chain hash.

| Param | Type | Description |
|-------|------|-------------|
| `action` | `str` | Action name. |
| `agent_name` | `str` | Agent performing the action. |
| `session_id` | `str` | Session identifier. |
| `user_id` | `str` | User identifier. |
| `correlation_id` | `str` | Correlation identifier. |
| `details` | `dict[str, Any] \| None` | Additional details. |
| `outcome` | `str` | Outcome label. Default `"success"`. |
| `duration_ms` | `float` | Duration in milliseconds. Default `0.0`. |

**Returns:** `AuditEntry`

Guardrail filters (`PermissionPolicy`, `RateLimitPolicy`, `ApprovalGate`) emit deterministic audit events when they are configured with an `AuditTrail`. Denials appear as `permission_denied` or `rate_limit_exceeded`, while approval flows write `approval_requested`, `approval_granted`, `approval_denied`, and `approval_timeout`, giving you a complete trail of governance decisions.

#### `query`

```python
async def query(
    self,
    session_id: str | None = None,
    agent_name: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[AuditEntry]
```

Query audit entries with optional filters.

**Returns:** `list[AuditEntry]`

#### `verify_chain`

```python
def verify_chain(self, entries: list[AuditEntry]) -> bool
```

Verify the hash chain integrity of a sequence of audit entries.

**Returns:** `bool` -- `True` if the chain is intact, `False` otherwise.

---

## ProfileEntry

Dataclass. A single profiling entry.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Operation label. |
| `duration_ms` | `float` | Duration in milliseconds. |
| `metadata` | `dict[str, Any]` | Arbitrary metadata. Default empty dict. |

---

## AgentProfiler

Tracks timing of agent operations via manual recording or context-manager measurement.

### Constructor

```python
def __init__(self) -> None
```

No parameters.

### Methods

#### `measure`

```python
def measure(self, label: str) -> _ProfileContext
```

Return a context manager that measures a block's wall-clock duration and records it.

| Param | Type | Description |
|-------|------|-------------|
| `label` | `str` | Label for the measured operation. |

**Returns:** Context manager (supports `with` statement).

#### `record`

```python
def record(self, label: str, duration_ms: float, **metadata: Any) -> None
```

Manually record a timing entry.

| Param | Type | Description |
|-------|------|-------------|
| `label` | `str` | Operation label. |
| `duration_ms` | `float` | Duration in milliseconds. |
| `**metadata` | `Any` | Arbitrary metadata attached to the entry. |

#### `total_ms`

```python
def total_ms(self) -> float
```

Return the sum of all recorded durations in milliseconds.

#### `summary`

```python
def summary(self) -> dict[str, Any]
```

Return a summary dict with `total_ms`, `entries` count, and `breakdown` by label (each with `count` and `total_ms`).

**Returns:** `dict[str, Any]`

#### `clear`

```python
def clear(self) -> None
```

Discard all recorded entries.

### Properties

#### `entries`

```python
@property
def entries(self) -> list[ProfileEntry]
```

Return a copy of all recorded profile entries.

---

## ReplayEntry

Dataclass. A single entry in a session recording.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Turn number (1-indexed). |
| `role` | `str` | Role of the speaker (e.g. `"user"`, `"assistant"`). |
| `content` | `str` | Message content. |
| `agent_name` | `str` | Agent that produced the message. Default `""`. |
| `timestamp` | `float` | Monotonic timestamp. Default `0.0`. |
| `metadata` | `dict[str, Any]` | Arbitrary metadata. Default empty dict. |

---

## SessionRecorder

Records full conversation turns for replay and debugging.

### Constructor

```python
def __init__(self) -> None
```

No parameters.

### Methods

#### `record_turn`

```python
def record_turn(self, role: str, content: str, agent_name: str = "", **metadata: Any) -> None
```

Record a single conversation turn.

| Param | Type | Description |
|-------|------|-------------|
| `role` | `str` | Speaker role. |
| `content` | `str` | Message content. |
| `agent_name` | `str` | Agent name. Default `""`. |
| `**metadata` | `Any` | Arbitrary metadata. |

#### `replay`

```python
def replay(self) -> list[ReplayEntry]
```

Return all recorded entries for replay.

#### `replay_from`

```python
def replay_from(self, turn: int) -> list[ReplayEntry]
```

Return entries starting from the given turn number.

| Param | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Turn number to start from (inclusive). |

**Returns:** `list[ReplayEntry]`

#### `to_dict`

```python
def to_dict(self) -> list[dict[str, Any]]
```

Serialize all entries to a list of plain dictionaries.

#### `clear`

```python
def clear(self) -> None
```

Discard all recorded entries and reset the turn counter.

### Properties

#### `turn_count`

```python
@property
def turn_count(self) -> int
```

Return the current turn number.

---

## TokenQuota

Dataclass. Tracks and enforces token usage limits within a sliding time window.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `max_tokens` | `int` | Maximum tokens allowed per window. Default `100_000`. |
| `window_seconds` | `float` | Sliding window duration in seconds. Default `3600.0`. |

### Methods

#### `record_usage`

```python
def record_usage(self, key: str, tokens: int) -> None
```

Record token usage for a key (user_id, session_id, etc.).

| Param | Type | Description |
|-------|------|-------------|
| `key` | `str` | Quota key. |
| `tokens` | `int` | Number of tokens consumed. |

#### `used`

```python
def used(self, key: str) -> int
```

Get total tokens used in the current window.

**Returns:** `int`

#### `remaining`

```python
def remaining(self, key: str) -> int
```

Get remaining tokens in quota (never negative).

**Returns:** `int`

#### `is_exceeded`

```python
def is_exceeded(self, key: str) -> bool
```

Check if quota is exceeded.

**Returns:** `bool`

---

## TokenQuotaFilter

Filter that enforces token quotas on agent inputs and outputs. Designed to be used as an agent filter.

### Constructor

```python
def __init__(
    self,
    quota: TokenQuota,
    key_field: str = "session_id",
    tokens_per_word: float = 1.3,
) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `quota` | `TokenQuota` | The quota to enforce. |
| `key_field` | `str` | Metadata field to use as the quota key. Default `"session_id"`. |
| `tokens_per_word` | `float` | Multiplier for word-to-token estimation. Default `1.3`. |

### Methods

#### `on_input`

```python
async def on_input(self, context: FilterContext) -> FilterResult
```

Deny input if quota is exceeded; otherwise record estimated token usage and allow.

**Returns:** `FilterResult` (`Allow` or `Deny`)

#### `on_output`

```python
async def on_output(self, context: FilterContext) -> FilterResult
```

Record estimated token usage for the output and allow.

**Returns:** `FilterResult`

#### `on_function_call`

```python
async def on_function_call(self, context: FilterContext) -> FilterResult
```

Always allows function calls (no quota check).

**Returns:** `FilterResult`

---

## instrument

```python
def instrument(exporter: Any = None, meter_provider: Any = None) -> None
```

Initialize OpenTelemetry tracing for SKTK. Idempotent and thread-safe. Subsequent calls are no-ops.

| Param | Type | Description |
|-------|------|-------------|
| `exporter` | `Any` | Optional span exporter to attach. |
| `meter_provider` | `Any` | Reserved for future use. |

---

## create_span

```python
@asynccontextmanager
async def create_span(
    name: str,
    attributes: dict[str, str] | None = None,
) -> AsyncIterator[trace.Span]
```

Async context manager that creates an OpenTelemetry span enriched with the current `ExecutionContext` (correlation_id, tenant_id, user_id, session_id).

| Param | Type | Description |
|-------|------|-------------|
| `name` | `str` | Span name. |
| `attributes` | `dict[str, str] \| None` | Additional span attributes. |

**Yields:** `trace.Span`
