# tests/unit/observability/test_tracing.py
import pytest

from sktk.core.context import ExecutionContext, context_scope
from sktk.observability.tracing import (
    _get_tracer,
    _reset_for_testing,
    create_span,
    instrument,
)


@pytest.fixture(autouse=True)
def reset_tracing():
    _reset_for_testing()
    yield
    _reset_for_testing()


def test_instrument_does_not_raise():
    instrument()


def test_instrument_idempotent():
    instrument()
    instrument()  # second call should be no-op


def test_instrument_rechecks_initialized_inside_lock(monkeypatch):
    import sktk.observability.tracing as tracing

    class FlipInitializedLock:
        def __enter__(self):
            tracing._initialized = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(tracing, "_initialized", False)
    monkeypatch.setattr(tracing, "_init_lock", FlipInitializedLock())
    tracing.instrument()
    assert tracing._initialized is True


def test_instrument_with_exporter():
    from unittest.mock import MagicMock

    exporter = MagicMock()
    instrument(exporter=exporter)


def test_get_tracer_returns_tracer():
    tracer = _get_tracer()
    assert tracer is not None


def test_get_tracer_lazy_init():
    """_get_tracer auto-instruments if not already initialized."""
    tracer = _get_tracer()
    assert tracer is not None


@pytest.mark.asyncio
async def test_create_span_context_manager():
    instrument()
    ctx = ExecutionContext(correlation_id="c1", tenant_id="t1")
    async with context_scope(ctx), create_span("test.operation") as span:
        assert span is not None


@pytest.mark.asyncio
async def test_create_span_with_all_context_fields():
    instrument()
    ctx = ExecutionContext(
        correlation_id="c2",
        tenant_id="t2",
        user_id="u2",
        session_id="s2",
    )
    async with context_scope(ctx), create_span("test.full", attributes={"custom": "val"}) as span:
        assert span is not None


@pytest.mark.asyncio
async def test_create_span_without_context():
    instrument()
    async with create_span("test.no_context") as span:
        assert span is not None
