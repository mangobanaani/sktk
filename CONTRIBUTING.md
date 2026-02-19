# Contributing to SKTK

Thank you for your interest in contributing to SKTK.

## Development Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/mangobanaani/sktk.git
cd sktk

# Install in development mode with all extras
pip install -e ".[dev,otel,redis,openai,openapi,mcp,a2a]"
```

### Running Tests

```bash
# Unit tests (no API keys needed)
pytest tests/unit/ -q

# With coverage
pytest tests/unit/ --cov=sktk --cov-report=term-missing

# Integration tests (requires ANTHROPIC_API_KEY)
pytest tests/integration/ -m integration

# End-to-end tests (requires ANTHROPIC_API_KEY)
pytest tests/e2e/ -m e2e
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Type checking
mypy src/sktk/ --strict

# Security scanning
bandit -r src/sktk/ -c pyproject.toml
```

## Code Standards

- **Python 3.11+** -- use modern syntax (type unions with `|`, match statements where appropriate)
- **Type annotations** -- all public functions must have complete type annotations. Use `Protocol` for interfaces.
- **Docstrings** -- all public classes and methods must have docstrings
- **Tests** -- all new code must have unit tests. Target 95% coverage.
- **Linting** -- code must pass `ruff check` and `mypy --strict`
- **No emojis** in code, comments, or commit messages

## Architecture

SKTK is organized into six packages:

| Package | Purpose |
|---------|---------|
| `sktk.agent` | Agent abstraction, providers, tools, filters, protocols |
| `sktk.core` | Context propagation, errors, types, resilience, configuration |
| `sktk.team` | Multi-agent orchestration, strategies, graph workflows |
| `sktk.knowledge` | RAG pipeline, vector backends, retrieval strategies |
| `sktk.session` | Conversation history, blackboard, session backends |
| `sktk.observability` | Tracing, events, metrics, audit, profiling |

### Key Design Principles

1. **Protocol-first** -- use `typing.Protocol` with `@runtime_checkable` for all interfaces
2. **Optional dependencies** -- guard imports with `try/except ImportError`, provide clear install instructions
3. **Immutable value types** -- use `frozen=True` dataclasses for data objects
4. **Composable filters** -- `Allow`/`Deny`/`Modify` result types with short-circuit evaluation
5. **Async-first** -- all I/O operations are async; use `maybe_await` for sync/async compatibility

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD encouraged)
3. Ensure all checks pass: `ruff check`, `mypy --strict`, `pytest`
4. Keep PRs focused -- one feature or fix per PR
5. Update CHANGELOG.md for user-facing changes
