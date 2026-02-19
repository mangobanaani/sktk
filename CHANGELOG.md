# Changelog

All notable changes to SKTK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-20

### Added

- Core agent abstraction (`SKTKAgent`) with fluent builder pattern
- Multi-LLM provider support (OpenAI, Azure OpenAI, Anthropic Claude, Google Gemini, local models)
- Provider router with fallback and round-robin strategies
- Agentic tool-calling loop with configurable max iterations
- Typed input/output contracts via Pydantic models with constrained decoding
- Guardrail filter pipeline with 9 built-in filters:
  - Content safety (regex-based blocking)
  - PII detection (email, phone, SSN, credit card, IP, IBAN, passport)
  - Token budget enforcement
  - Prompt injection detection (with Unicode normalization)
  - Permission policies (allowlist/denylist)
  - Rate limiting (per-user/session)
  - Approval gate (human-in-the-loop)
  - Memory grounding (auto-inject relevant context)
  - Content safety for streaming (per-chunk filtering)
- Session management with pluggable backends (in-memory, SQLite, Redis)
- Conversation history with summarization and token budget management
- Blackboard (shared key-value state) with typed access and watchers
- Knowledge/RAG pipeline:
  - Text chunking with configurable strategies
  - Three vector backends (in-memory, FAISS, HNSW)
  - Dense, sparse (BM25), and hybrid retrieval with Reciprocal Rank Fusion
  - Grounding filter for automatic RAG injection
- Multi-agent orchestration:
  - Team coordination strategies (round-robin, capability-based, broadcast, priority)
  - Pipeline DSL with `>>` operator
  - Parallel fan-out/fan-in execution
  - Topology builder with Mermaid diagram generation
- Graph-based workflows with conditional edges, checkpointing, and loop protection
- Observability stack:
  - OpenTelemetry integration (optional dependency)
  - Structured event stream with pluggable sinks
  - Token quota tracking and enforcement
  - Agent profiler with timing measurements
  - Tamper-evident audit trail with SHA-256 hash chain verification
- Protocol support:
  - MCP client (consume tools from MCP servers)
  - MCP server (expose agent as MCP endpoint)
  - A2A protocol (agent discovery and invocation via JSON-RPC)
  - OpenAPI tool generation (with real HTTP calls via httpx)
- Semantic memory (cross-session remember/recall/forget backed by knowledge base)
- Multimodal message types (text, image, document, tool result)
- Prompt optimization via iterative hill-climbing on test suites
- Comprehensive testing toolkit:
  - MockKernel for deterministic testing without LLMs
  - PluginSandbox for isolated tool testing
  - PromptSuite for regression testing
  - Assertion helpers (history, events, blackboard)
- Middleware stack for cross-cutting concerns
- Configuration management via YAML/JSON/environment
- Secrets management with env, file, and chained providers
- Circuit breaker and retry policies with configurable backoff
- Thread pool executor for CPU-bound operations
- Lifecycle hooks (pre/post invoke, on error)

### Security

- Prompt injection detection with Unicode normalization and zero-width character stripping
- PII detection covering email, phone, SSN, credit card, IP address, IBAN, and passport patterns
- Permission-based tool access control
- Human-in-the-loop approval gate with timeout
- Tamper-evident audit trail
