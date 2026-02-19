# SKTK API Reference

Semantic Kernel Toolkit -- a batteries-included layer over Microsoft Semantic Kernel for building production-grade LLM agents in Python.

## Packages

| Package | Description |
|---------|-------------|
| [`sktk.agent`](agent.md) | Agent creation, tools, filters, contracts, hooks, middleware, capabilities, providers, planning, templates |
| [`sktk.core`](core.md) | Context propagation, events, types, errors, resilience (retry/circuit-breaker), executor, secrets, config |
| [`sktk.knowledge`](knowledge.md) | RAG pipeline: knowledge bases, chunking, BM25/dense retrieval, reciprocal rank fusion, grounding |
| [`sktk.session`](session.md) | Conversation history, blackboard shared state, persistence backends (memory, SQLite, Redis), summarizers |
| [`sktk.team`](team.md) | Multi-agent orchestration: teams, strategies (round-robin, broadcast, capability routing), topology DSL |
| [`sktk.observability`](observability.md) | Event streaming, structured logging, token tracking/pricing, audit trail, profiler, quota enforcement, tracing |
| [`sktk.testing`](testing.md) | Test infrastructure: mock kernels, LLM scenarios, assertions, fixtures, plugin sandbox, prompt regression suites |

## Quick Import Guide

Most symbols are re-exported from the root `sktk` package:

```python
from sktk import SKTKAgent, Session, Tool, tool

# Filters
from sktk import PIIFilter, PromptInjectionFilter, ContentSafetyFilter

# Events
from sktk import MessageEvent, CompletionEvent, ToolCallEvent

# Resilience
from sktk import RetryPolicy, BackoffStrategy, CircuitBreaker

# Testing (no LLM needed)
agent = SKTKAgent.with_responses("bot", ["scripted reply"])

# Knowledge / RAG
from sktk import KnowledgeBase, TextSource, InMemoryKnowledgeBackend

# Session persistence
from sktk import SQLiteHistory, InMemoryHistory, InMemoryBlackboard

# Multi-agent
from sktk import SKTKTeam, RoundRobinStrategy, BroadcastStrategy

# Observability
from sktk import TokenTracker, AuditTrail, AgentProfiler
```

## Examples

For a guided learning path and run instructions, see [`examples/README.md`](../../examples/README.md).

Operational runbooks live in [`docs/ops/README.md`](../ops/README.md).

### Getting Started

| Example | What it shows |
|---------|---------------|
| [`01_basic_agent.py`](../../examples/getting_started/01_basic_agent.py) | Create an agent and invoke it with scripted responses |
| [`02_persistent_session.py`](../../examples/getting_started/02_persistent_session.py) | SQLite-backed history across restarts |
| [`03_tools_and_contracts.py`](../../examples/getting_started/03_tools_and_contracts.py) | `@tool` decorator, schema introspection, typed Pydantic output contracts |
| [`04_lifecycle_hooks.py`](../../examples/getting_started/04_lifecycle_hooks.py) | Lifecycle hooks (start/complete/error) and middleware stacks |

### Concepts: Agent

| Example | What it shows |
|---------|---------------|
| [`agent/provider_router_fallback.py`](../../examples/concepts/agent/provider_router_fallback.py) | Provider registry setup and router fallback when primary fails |
| [`agent/approval_permissions_rate_limits.py`](../../examples/concepts/agent/approval_permissions_rate_limits.py) | Permission allowlists, human approval gates, and invocation rate limits |
| [`agent/openapi_tools_end_to_end.py`](../../examples/concepts/agent/openapi_tools_end_to_end.py) | Generate tools from OpenAPI specs and call tool stubs end-to-end |

### Concepts: Knowledge

| Example | What it shows |
|---------|---------------|
| [`knowledge/rag_with_chunking.py`](../../examples/concepts/knowledge/rag_with_chunking.py) | End-to-end RAG: ingest, chunk, index, query |
| [`knowledge/retrieval_modes_comparison.py`](../../examples/concepts/knowledge/retrieval_modes_comparison.py) | Side-by-side dense/sparse/hybrid retrieval ranking behavior |

### Concepts: Multi-Agent

| Example | What it shows |
|---------|---------------|
| [`multi_agent/team_with_round_robin.py`](../../examples/concepts/multi_agent/team_with_round_robin.py) | SKTKTeam with round-robin strategy and event streaming |
| [`multi_agent/orchestration_patterns.py`](../../examples/concepts/multi_agent/orchestration_patterns.py) | Pattern index/runner for focused orchestration demos |
| [`multi_agent/pipeline_topology.py`](../../examples/concepts/multi_agent/pipeline_topology.py) | Pipeline DSL with `>>` operator, parallel/sequential nodes |
| [`multi_agent/guardrails_and_providers.py`](../../examples/concepts/multi_agent/guardrails_and_providers.py) | Input/output filters and provider registry |
| [`multi_agent/custom_strategy.py`](../../examples/concepts/multi_agent/custom_strategy.py) | Writing a custom orchestration strategy |

### Concepts: Multi-Agent Patterns

| Example | What it shows |
|---------|---------------|
| [`multi_agent/patterns/01_sequential_pipeline.py`](../../examples/concepts/multi_agent/patterns/01_sequential_pipeline.py) | Chain-of-responsibility execution across agents |
| [`multi_agent/patterns/02_parallel_fanout_fanin.py`](../../examples/concepts/multi_agent/patterns/02_parallel_fanout_fanin.py) | Parallel fan-out workers merged by a synthesizer |
| [`multi_agent/patterns/03_supervisor_worker.py`](../../examples/concepts/multi_agent/patterns/03_supervisor_worker.py) | Supervisor style coordination over specialized workers |
| [`multi_agent/patterns/04_reflection_loop.py`](../../examples/concepts/multi_agent/patterns/04_reflection_loop.py) | Generator/critic refinement loop |
| [`multi_agent/patterns/05_debate_consensus.py`](../../examples/concepts/multi_agent/patterns/05_debate_consensus.py) | Debate between opposing agents plus consensus judge |

### Concepts: Session, Resilience, Observability, Testing

| Example | What it shows |
|---------|---------------|
| [`session/blackboard_shared_state.py`](../../examples/concepts/session/blackboard_shared_state.py) | Typed key-value blackboard for inter-agent coordination |
| [`resilience/resilience_patterns.py`](../../examples/concepts/resilience/resilience_patterns.py) | RetryPolicy with exponential jitter, circuit breaker state machine |
| [`observability/observability_stack.py`](../../examples/concepts/observability/observability_stack.py) | Token cost tracking, tamper-evident audit trail, profiling |
| [`observability/event_stream_logging.py`](../../examples/concepts/observability/event_stream_logging.py) | EventStream sinks forwarding runtime events to structured logs |
| [`testing/testing_patterns.py`](../../examples/concepts/testing/testing_patterns.py) | MockKernel, PluginSandbox, PromptSuite regression testing |
