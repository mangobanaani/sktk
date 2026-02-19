.PHONY: all lint format test coverage typecheck security examples benchmark check clean

all: lint test

lint:
	@echo "--- ruff check ---"
	ruff check src/ tests/
	@echo "--- ruff format check ---"
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/unit/ -q

test-all:
	pytest tests/ -q

coverage:
	pytest tests/unit/ --cov=sktk --cov-report=term-missing --cov-fail-under=95

typecheck:
	mypy --strict src/sktk/

security:
	@echo "--- bandit ---"
	bandit -r src/sktk/ -c pyproject.toml -q

examples:
	@set -e; for f in \
		examples/getting_started/01_basic_agent.py \
		examples/getting_started/02_persistent_session.py \
		examples/getting_started/03_tools_and_contracts.py \
		examples/getting_started/04_lifecycle_hooks.py \
		examples/concepts/agent/provider_router_fallback.py \
		examples/concepts/agent/approval_permissions_rate_limits.py \
		examples/concepts/agent/openapi_tools_end_to_end.py \
		examples/concepts/session/blackboard_shared_state.py \
		examples/concepts/knowledge/rag_with_chunking.py \
		examples/concepts/knowledge/retrieval_modes_comparison.py \
		examples/concepts/resilience/resilience_patterns.py \
		examples/concepts/observability/observability_stack.py \
		examples/concepts/observability/event_stream_logging.py \
		examples/concepts/testing/testing_patterns.py \
		examples/concepts/multi_agent/team_with_round_robin.py \
		examples/concepts/multi_agent/pipeline_topology.py \
		examples/concepts/multi_agent/custom_strategy.py \
		examples/concepts/multi_agent/guardrails_and_providers.py \
		examples/concepts/multi_agent/orchestration_patterns.py; \
	do echo "Running $$f"; python "$$f"; done

benchmark:
	python benchmarks/agent_runtime_benchmark.py \
		--iterations 400 --warmup 40 --max-p95-ms 5.0 --min-qps 10000

check: lint typecheck test security

clean:
	rm -rf .pytest_cache .coverage .mypy_cache htmlcov *.egg-info dist build .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
