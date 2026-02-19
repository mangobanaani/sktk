# sktk.session

Session state management: conversation history, shared blackboard, persistence, and summarization.

## Table of Contents

- [Session](#session)
- [ConversationHistory (abstract)](#conversationhistory)
- [Blackboard (abstract)](#blackboard)
- [Backends](#backends)
  - [InMemoryHistory](#inmemoryhistory)
  - [InMemoryBlackboard](#inmemoryblackboard)
  - [SQLiteHistory](#sqlitehistory)
  - [RedisHistory](#redishistory)
- [Summarizers](#summarizers)
  - [SummaryResult](#summaryresult)
  - [WindowSummarizer](#windowsummarizer)
  - [TokenBudgetSummarizer](#tokenbudgetsummarizer)

---

## Session

```python
@dataclass
class Session
```

The unit of shared state for multi-agent interactions. Supports use as an async context manager (`async with Session(...) as s:`).

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(required)* | Unique session identifier. |
| `tenant_id` | `str \| None` | `None` | Optional tenant/organization scope. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary session-level metadata. |
| `history` | `ConversationHistory` | `InMemoryHistory()` | Conversation history backend. |
| `blackboard` | `Blackboard` | `InMemoryBlackboard()` | Shared agent memory backend. |

### Methods

#### close

```python
async def close() -> None
```

Explicitly close session resources. Calls `close()` on the history backend if it exposes one.

#### Async Context Manager

```python
async with Session(id="s1") as session:
    ...
```

On exit, calls `close()` on the history backend if available.

---

## ConversationHistory

```python
class ConversationHistory(ABC)
```

Abstract base class for conversation history storage. All methods are abstract and async.

### Methods (all abstract)

#### append

```python
@abstractmethod
async def append(role: str, content: str, metadata: dict | None = None) -> None
```

Add a message to the conversation history.

| Param | Type | Default | Description |
|---|---|---|---|
| `role` | `str` | *(required)* | Message role (e.g. `"user"`, `"assistant"`, `"system"`). |
| `content` | `str` | *(required)* | Message text content. |
| `metadata` | `dict \| None` | `None` | Optional metadata attached to the message. |

#### get

```python
@abstractmethod
async def get(limit: int | None = None, roles: list[str] | None = None) -> list[dict]
```

Return messages, optionally filtered by role and limited to the N most recent.

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | `int \| None` | `None` | If set, return only the last N messages (after filtering). |
| `roles` | `list[str] \| None` | `None` | If set, include only messages with these roles. |

**Returns:** `list[dict]` -- each dict has keys `"role"`, `"content"`, `"metadata"`.

#### clear

```python
@abstractmethod
async def clear() -> None
```

Remove all messages from this history.

#### fork

```python
@abstractmethod
async def fork(session_id: str) -> ConversationHistory
```

Create an independent copy of this history under a new session ID.

| Param | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID for the forked copy. |

**Returns:** `ConversationHistory`

#### \_\_len\_\_

```python
@abstractmethod
def __len__() -> int
```

Return the number of messages in this history.

---

## Blackboard

```python
class Blackboard(ABC)
```

Abstract base class for shared agent memory (key-value store with Pydantic model values). All methods are abstract and async.

### Methods (all abstract)

#### set

```python
@abstractmethod
async def set(key: str, value: BaseModel) -> None
```

Store a Pydantic model under the given key.

| Param | Type | Description |
|---|---|---|
| `key` | `str` | Storage key. |
| `value` | `BaseModel` | Pydantic model instance to store. |

#### get

```python
@abstractmethod
async def get(key: str, model: type[T]) -> T | None
```

Retrieve and validate a value by key, returning `None` if absent.

| Param | Type | Description |
|---|---|---|
| `key` | `str` | Storage key. |
| `model` | `type[T]` | Pydantic model class to validate the stored value against. `T` is bound to `BaseModel`. |

**Returns:** `T | None`

#### get_all

```python
@abstractmethod
async def get_all(prefix: str) -> dict[str, Any]
```

Return all key-value pairs whose keys start with `prefix`.

| Param | Type | Description |
|---|---|---|
| `prefix` | `str` | Key prefix to filter by. |

**Returns:** `dict[str, Any]`

#### delete

```python
@abstractmethod
async def delete(key: str) -> bool
```

Delete a key and return `True` if it existed.

| Param | Type | Description |
|---|---|---|
| `key` | `str` | Key to delete. |

**Returns:** `bool`

#### watch

```python
@abstractmethod
async def watch(key: str) -> AsyncIterator[BaseModel]
```

Yield new values for a key as they are set. This is an async generator.

| Param | Type | Description |
|---|---|---|
| `key` | `str` | Key to watch. |

**Yields:** `BaseModel`

#### keys

```python
@abstractmethod
async def keys(prefix: str = "") -> list[str]
```

List all keys, optionally filtered by prefix.

| Param | Type | Default | Description |
|---|---|---|---|
| `prefix` | `str` | `""` | Key prefix to filter by. Empty string matches all. |

**Returns:** `list[str]`

---

## Backends

### InMemoryHistory

```python
class InMemoryHistory(ConversationHistory)
```

In-memory conversation history backed by a plain list. Thread-safe via `asyncio.Lock`.

#### Constructor

```python
InMemoryHistory()
```

No parameters.

#### Methods

Implements all `ConversationHistory` abstract methods:

| Method | Notes |
|---|---|
| `append(role, content, metadata)` | Appends to the in-memory list. |
| `get(limit, roles)` | Returns deep-copied messages. |
| `clear()` | Removes all messages. |
| `fork(session_id)` | Returns a new `InMemoryHistory` with deep-copied messages. |
| `__len__()` | Returns message count. |

---

### InMemoryBlackboard

```python
class InMemoryBlackboard(Blackboard)
```

In-memory blackboard storing serialized Pydantic models as JSON strings. Thread-safe via `asyncio.Lock`.

#### Constructor

```python
InMemoryBlackboard()
```

No parameters.

#### Methods

Implements all `Blackboard` abstract methods:

| Method | Notes |
|---|---|
| `set(key, value)` | Stores the value and notifies any active watchers. |
| `get(key, model)` | Validates on retrieval; raises `BlackboardTypeError` on type mismatch. |
| `get_all(prefix)` | Returns entries as parsed dicts. |
| `delete(key)` | Returns `True` if key existed. |
| `watch(key)` | Uses per-watcher `asyncio.Queue` internally. |
| `keys(prefix)` | Lists matching keys. |

---

### SQLiteHistory

```python
class SQLiteHistory(ConversationHistory)
```

SQLite-backed persistent conversation history. All blocking SQLite calls are offloaded to a thread via `asyncio.to_thread`.

#### Constructor

```python
SQLiteHistory(db_path: str, session_id: str)
```

| Param | Type | Description |
|---|---|---|
| `db_path` | `str` | Path to the SQLite database file. |
| `session_id` | `str` | Session identifier used to scope messages. |

#### Methods

##### initialize

```python
async def initialize() -> None
```

Create the `messages` table if needed and count existing rows. Must be called before using any other method.

##### close

```python
async def close() -> None
```

Close the SQLite connection.

##### append

```python
async def append(role: str, content: str, metadata: dict | None = None) -> None
```

Insert a message row, offloading the blocking SQL write to a thread.

##### get

```python
async def get(limit: int | None = None, roles: list[str] | None = None) -> list[dict]
```

Query messages, optionally filtered by role and limited to the N most recent.

##### clear

```python
async def clear() -> None
```

Delete all messages for this session.

##### fork

```python
async def fork(session_id: str) -> SQLiteHistory
```

Copy all messages into a new session via `INSERT ... SELECT`. Returns a new initialized `SQLiteHistory`.

| Param | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID for the forked copy. |

**Returns:** `SQLiteHistory`

##### \_\_len\_\_

```python
def __len__() -> int
```

Return the cached message count.

---

### RedisHistory

```python
class RedisHistory(ConversationHistory)
```

Redis-backed persistent conversation history. Requires the `redis` extra: `pip install skat[redis]`. The Redis client is lazily initialized on first use.

#### Constructor

```python
RedisHistory(url: str = "redis://localhost:6379", session_id: str = "")
```

| Param | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | `"redis://localhost:6379"` | Redis connection URL. |
| `session_id` | `str` | `""` | Session identifier. Messages are stored under key `skat:history:{session_id}`. |

#### Methods

##### append

```python
async def append(role: str, content: str, metadata: dict | None = None) -> None
```

Append a message to the Redis list.

##### get

```python
async def get(limit: int | None = None, roles: list[str] | None = None) -> list[dict]
```

Retrieve messages, optionally filtered by role and limited to the N most recent.

##### clear

```python
async def clear() -> None
```

Delete all messages for this session.

##### fork

```python
async def fork(session_id: str) -> RedisHistory
```

Copy all messages into a new `RedisHistory` under a different session ID.

| Param | Type | Description |
|---|---|---|
| `session_id` | `str` | Session ID for the forked copy. |

**Returns:** `RedisHistory`

##### close

```python
async def close() -> None
```

Close the underlying Redis connection.

##### \_\_len\_\_

```python
def __len__() -> int
```

Return the cached message count.

---

## Summarizers

### SummaryResult

```python
@dataclass
class SummaryResult
```

Result of summarizing a conversation.

#### Fields

| Field | Type | Description |
|---|---|---|
| `messages` | `list[dict[str, Any]]` | The resulting message list after summarization. |
| `original_count` | `int` | Number of messages before summarization. |
| `summarized_count` | `int` | Number of messages after summarization. |
| `summary_text` | `str` | The generated summary text (empty string if no summarization occurred). |

---

### WindowSummarizer

```python
class WindowSummarizer
```

Keep only the most recent N messages, with an optional system summary. System messages are preserved separately when `keep_system=True`. Dropped messages are replaced with a single system message noting the count.

#### Constructor

```python
WindowSummarizer(window_size: int = 20, keep_system: bool = True)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `window_size` | `int` | `20` | Maximum number of non-system messages to keep. |
| `keep_system` | `bool` | `True` | Whether to preserve system messages outside the window. |

#### Methods

##### summarize

```python
def summarize(messages: list[dict[str, Any]]) -> SummaryResult
```

Summarize the conversation by truncating to the window size. If `len(messages) <= window_size`, returns messages unchanged.

| Param | Type | Description |
|---|---|---|
| `messages` | `list[dict[str, Any]]` | Full conversation message list. Each dict has `"role"` and `"content"` keys. |

**Returns:** `SummaryResult`

---

### TokenBudgetSummarizer

```python
class TokenBudgetSummarizer
```

Keep messages within a token budget using word-count estimation. Estimates tokens as `words * tokens_per_word`. Trims oldest non-system messages first. Reserves 50 tokens for the summary message itself.

#### Constructor

```python
TokenBudgetSummarizer(
    max_tokens: int = 4000,
    tokens_per_word: float = 1.3,
    keep_system: bool = True,
)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | `4000` | Maximum token budget for the output messages. |
| `tokens_per_word` | `float` | `1.3` | Multiplier to estimate tokens from word count. |
| `keep_system` | `bool` | `True` | Whether to preserve system messages outside the budget. |

#### Methods

##### summarize

```python
def summarize(messages: list[dict[str, Any]]) -> SummaryResult
```

Summarize the conversation by trimming to fit within the token budget. If estimated token count is already within budget, returns messages unchanged.

| Param | Type | Description |
|---|---|---|
| `messages` | `list[dict[str, Any]]` | Full conversation message list. Each dict has `"role"` and `"content"` keys. |

**Returns:** `SummaryResult`
