# sktk.team

Multi-agent orchestration with pluggable coordination strategies and pipeline topology.

## Table of Contents

- [SKTKTeam](#sktkteam)
- [CoordinationStrategy (Protocol)](#coordinationstrategy-protocol)
- [RoundRobinStrategy](#roundrobinstrategy)
- [BroadcastStrategy](#broadcaststrategy)
- [CapabilityRoutingStrategy](#capabilityroutingstrategy)
- [ComposedStrategy](#composedstrategy)
- [CapabilityRouter](#capabilityrouter)
- [AgentNode](#agentnode)
- [SequentialNode](#sequentialnode)
- [ParallelNode](#parallelnode)
- [TopologyNode](#topologynode)

---

## SKTKTeam

Dataclass. Orchestrates a list of agents using a pluggable coordination strategy.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `agents` | `list[SKTKAgent]` | Agents participating in the team. |
| `strategy` | `Any` | Coordination strategy (must conform to `CoordinationStrategy`). |
| `session` | `Session \| None` | Optional session providing conversation history. Default `None`. |
| `max_rounds` | `int` | Maximum number of sequential agent invocations. Default `20`. |

### Methods

#### `run`

```python
async def run(self, task: str, **kwargs: Any) -> Any
```

Execute the team on a task using the configured strategy. Delegates to broadcast or sequential execution depending on strategy type.

| Param | Type | Description |
|-------|------|-------------|
| `task` | `str` | The task or message to process. |
| `**kwargs` | `Any` | Forwarded to the strategy's `next_agent`. |

**Returns:** `Any` -- final result from the last agent (sequential) or `list[Any]` of all results (broadcast).

#### `stream`

```python
async def stream(self, task: str, **kwargs: Any) -> AsyncIterator[Any]
```

Yield `MessageEvent` instances as each agent completes, ending with a `CompletionEvent`.

| Param | Type | Description |
|-------|------|-------------|
| `task` | `str` | The task or message to process. |
| `**kwargs` | `Any` | Forwarded to the strategy's `next_agent`. |

**Returns:** `AsyncIterator[Any]` -- stream of `MessageEvent` and a final `CompletionEvent`.

---

## CoordinationStrategy (Protocol)

```python
@runtime_checkable
class CoordinationStrategy(Protocol)
```

Protocol that all coordination strategies must satisfy.

### Methods

#### `next_agent`

```python
async def next_agent(
    self,
    agents: list[SKTKAgent],
    history: ConversationHistory | None,
    task: str,
    **kwargs: Any,
) -> SKTKAgent | None
```

Select the next agent to handle the task, or return `None` to stop.

---

## RoundRobinStrategy

Cycles through agents in order, one per call.

### Constructor

```python
def __init__(self) -> None
```

No parameters. Internal index starts at 0.

### Methods

#### `next_agent`

```python
async def next_agent(
    self,
    agents: list[Any],
    history: ConversationHistory | None,
    task: str,
    **kwargs: Any,
) -> Any | None
```

Return the next agent in round-robin order. Returns `None` if `agents` is empty.

#### `reset`

```python
def reset(self) -> None
```

Reset the internal index to 0.

#### `__or__`

```python
def __or__(self, other: Any) -> ComposedStrategy
```

Compose this strategy with another using the `|` operator.

---

## BroadcastStrategy

Sends the task to all agents in parallel.

### Constructor

```python
# No parameters
```

### Methods

#### `next_agent`

```python
async def next_agent(
    self,
    agents: list[Any],
    history: ConversationHistory,
    task: str,
    **kwargs: Any,
) -> None
```

Always returns `None`; broadcast delegates to `get_all_agents` instead.

#### `get_all_agents`

```python
def get_all_agents(self, agents: list[Any]) -> list[Any]
```

Return a copy of the full agent list for parallel dispatch.

#### `__or__`

```python
def __or__(self, other: Any) -> ComposedStrategy
```

Compose this strategy with another using the `|` operator.

---

## CapabilityRoutingStrategy

Routes to the first agent whose capabilities match the request.

### Constructor

```python
# No parameters
```

### Methods

#### `next_agent`

```python
async def next_agent(
    self,
    agents: list[Any],
    history: ConversationHistory | None,
    task: str,
    *,
    input_type: type[BaseModel] | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any | None
```

Return the first agent matching the given `input_type` or `tags`. Returns `None` if no match is found or both filters are `None`.

| Param | Type | Description |
|-------|------|-------------|
| `agents` | `list[Any]` | Candidate agents. |
| `history` | `ConversationHistory \| None` | Conversation history (unused). |
| `task` | `str` | Current task text. |
| `input_type` | `type[BaseModel] \| None` | Pydantic model type to match against agent capabilities. |
| `tags` | `list[str] \| None` | Tags to match against agent capabilities. |

#### `__or__`

```python
def __or__(self, other: Any) -> ComposedStrategy
```

Compose this strategy with another using the `|` operator.

---

## ComposedStrategy

Tries multiple strategies in order, returning the first match.

### Constructor

```python
def __init__(self, strategies: list[Any]) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `strategies` | `list[Any]` | Ordered list of strategies to try. |

### Methods

#### `next_agent`

```python
async def next_agent(
    self,
    agents: list[Any],
    history: ConversationHistory | None,
    task: str,
    **kwargs: Any,
) -> Any | None
```

Delegate to each strategy in order until one returns a non-`None` agent.

#### `__or__`

```python
def __or__(self, other: Any) -> ComposedStrategy
```

Append another strategy to the composition chain.

---

## CapabilityRouter

Capability-based task routing. Routes a request to the first agent whose capabilities match.

### Constructor

```python
def __init__(self, agents: list[SKTKAgent]) -> None
```

| Param | Type | Description |
|-------|------|-------------|
| `agents` | `list[SKTKAgent]` | Pool of agents to route among. |

### Methods

#### `route`

```python
def route(
    self,
    *,
    input_type: type[BaseModel] | None = None,
    tags: list[str] | None = None,
) -> SKTKAgent
```

Return the first agent whose capabilities match the given type or tags.

| Param | Type | Description |
|-------|------|-------------|
| `input_type` | `type[BaseModel] \| None` | Pydantic model type to match. |
| `tags` | `list[str] \| None` | Tags to match. |

**Returns:** `SKTKAgent`

**Raises:** `NoCapableAgentError` if no agent matches.

---

## AgentNode

Dataclass. Leaf node wrapping a single agent in a topology pipeline.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent` | `SKTKAgent` | The wrapped agent. |

### Methods

#### `__rshift__`

```python
def __rshift__(self, other: Any) -> SequentialNode
```

The `>>` operator. Chain this node with another node, agent, or list (parallel fan-out).

#### `run`

```python
async def run(self, message: str, **kwargs: Any) -> Any
```

Invoke the wrapped agent with the given message.

#### `visualize`

```python
def visualize(self) -> str
```

Return a Mermaid-formatted label for this agent node.

---

## SequentialNode

Dataclass. Chain two nodes in sequence: left output feeds right input.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `left` | `Any` | Left (upstream) node. |
| `right` | `Any` | Right (downstream) node. |

### Methods

#### `__rshift__`

```python
def __rshift__(self, other: Any) -> SequentialNode
```

The `>>` operator. Append another node to the pipeline.

#### `run`

```python
async def run(self, message: str, **kwargs: Any) -> Any
```

Run the left node, then feed its string output into the right node.

#### `visualize`

```python
def visualize(self) -> str
```

Return a Mermaid `graph LR` definition for this pipeline.

---

## ParallelNode

Dataclass. Fan-out: run all child nodes in parallel and collect results.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `list[Any]` | Child nodes to execute concurrently. |

### Methods

#### `__rshift__`

```python
def __rshift__(self, other: Any) -> SequentialNode
```

The `>>` operator. Pipe the parallel results into a downstream node.

#### `run`

```python
async def run(self, message: str, **kwargs: Any) -> list[Any]
```

Run all child nodes concurrently and return their results.

**Returns:** `list[Any]`

---

## TopologyNode

Factory for creating topology nodes.

### Static Methods

#### `from_agent`

```python
@staticmethod
def from_agent(agent: SKTKAgent) -> AgentNode
```

Wrap an agent in an `AgentNode`.

| Param | Type | Description |
|-------|------|-------------|
| `agent` | `SKTKAgent` | Agent to wrap. |

**Returns:** `AgentNode`
