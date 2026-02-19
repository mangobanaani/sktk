# SKTK Architecture Decision Records

## Why contextvars for context propagation

**Decision:** Use Python's `contextvars` module for execution context propagation.

**Alternatives considered:**
- Thread-local storage: Does not work with async code; multiple coroutines share a thread.
- Explicit parameter passing: Requires threading a context object through every function signature in every layer. Scales badly.
- Global state: No isolation between concurrent requests.

**Rationale:** `contextvars` provides copy-on-write semantics per task. Each `asyncio.Task` gets its own context snapshot. This means concurrent agent invocations within the same process are fully isolated without any coordination. The one gotcha -- `asyncio.create_task()` does not propagate context automatically -- is solved by the `propagate_context()` decorator.

## Why Pydantic v2 throughout

**Decision:** All data models, contracts, and configuration use Pydantic v2 `BaseModel`.

**Alternatives considered:**
- dataclasses: No runtime validation, no JSON schema generation, no serialization.
- TypedDict: Structural typing only, no validation at runtime.
- attrs: Good library, but Pydantic v2 has superior JSON handling and is already a dependency of Semantic Kernel.

**Rationale:** In multi-agent systems, the boundary between agents is where bugs concentrate. Typed contracts with runtime validation catch serialization errors, schema mismatches, and malformed LLM output at the boundary rather than deep inside agent logic. Pydantic v2's performance is acceptable for this use case.

## Why async-first

**Decision:** All SKTK APIs are async. No synchronous wrappers.

**Alternatives considered:**
- Sync-first with `asyncio.run()` escape hatch: Simpler API for simple cases.
- Dual sync/async API: Doubles the API surface and maintenance burden.

**Rationale:** Semantic Kernel's Python SDK is async-first. LLM calls are I/O-bound. Multi-agent orchestration involves concurrent agent execution. Async is not optional for production multi-agent systems. Providing sync wrappers would be misleading about the execution model.

## Why typed inter-agent contracts

**Decision:** Agents declare typed input/output contracts as Pydantic models.

**Alternatives considered:**
- String-only message passing: Simple, but errors are discovered only at runtime when an agent receives malformed input.
- JSON schema validation: Possible, but Pydantic provides better ergonomics and error messages.

**Rationale:** In a team of agents written by different developers, the producer and consumer of a message may have different assumptions about its schema. Typed contracts make these assumptions explicit and validated. When an LLM produces output that doesn't match the contract, SKTK raises a `ContractValidationError` that includes the raw output for debugging.

## Why the >> operator for topologies

**Decision:** Pipeline topologies are defined using Python's `>>` operator.

**Alternatives considered:**
- Imperative builder pattern: `Pipeline().add(a).add(b).parallel(c, d).add(e)`. Verbose and hard to read at a glance.
- YAML/JSON configuration: External configuration, loses IDE support and type checking.
- Decorator-based: `@pipeline(after=a)` -- ties topology to agent definition.

**Rationale:** `a >> b >> [c, d] >> e` reads like a data flow diagram. The topology is visible at a glance. The implementation uses `__rshift__` overloading to build a `TopologyNode` tree, which can be inspected, serialized, and visualized as a Mermaid diagram.

## Why capability declarations are structured types

**Decision:** Agent capabilities are declared as `Capability` dataclasses with typed `input_types` and `output_types`.

**Alternatives considered:**
- Prompt-based capability discovery: Ask the LLM "what can you do?" -- unreliable and non-deterministic.
- String tags only: Simple, but no type-safe routing.

**Rationale:** Structured capability matching allows the router to match task input type against agent input_contract types without any LLM call. This is deterministic, fast, and testable. Tags provide a secondary matching dimension for semantic grouping.

## Why wrap SK rather than replace it

**Decision:** SKTK wraps Semantic Kernel's primitives. All SK objects remain accessible.

**Alternatives considered:**
- Replace SK with custom implementation: Full control but massive scope and no ecosystem.
- Thin utility layer: Too little value-add.

**Rationale:** SK provides solid LLM connectors, plugin system, function calling, and planners. SKTK adds what SK lacks: persistent sessions, shared memory, typed contracts, guardrails, and composable orchestration. The wrapping is transparent -- users can access the underlying SK `Kernel`, `ChatCompletionAgent`, and orchestration classes when they need to.
