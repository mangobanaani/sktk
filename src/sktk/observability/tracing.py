"""OpenTelemetry instrumentation for SKTK operations."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sktk.core.context import get_context

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

_tracer: Any = None
_initialized = False
_init_lock = threading.Lock()


class _NoOpSpan:
    """Minimal stand-in when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


def instrument(exporter: Any = None) -> None:
    """Initialize OpenTelemetry tracing for SKTK. Idempotent and thread-safe."""
    global _tracer, _initialized
    if not _HAS_OTEL:
        _initialized = True
        return
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        # Only create our own provider if an exporter is requested;
        # otherwise use whatever provider is already set (which the
        # user may have configured externally).
        if exporter:
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("skat", "0.1.0")
        _initialized = True


def _get_tracer() -> Any:
    """Return the module-level tracer, lazily initializing if needed."""
    global _tracer
    if _tracer is None:
        instrument()
    return _tracer


@asynccontextmanager
async def create_span(name: str, attributes: dict[str, str] | None = None) -> AsyncIterator[Any]:
    """Create an OpenTelemetry span enriched with execution context."""
    tracer = _get_tracer()
    ctx = get_context()

    span_attributes: dict[str, str] = {}
    if ctx:
        span_attributes["sktk.correlation_id"] = ctx.correlation_id
        if ctx.tenant_id:
            span_attributes["sktk.tenant_id"] = ctx.tenant_id
        if ctx.user_id:
            span_attributes["sktk.user_id"] = ctx.user_id
        if ctx.session_id:
            span_attributes["sktk.session_id"] = ctx.session_id
    if attributes:
        span_attributes.update(attributes)

    if tracer is not None and _HAS_OTEL:
        with tracer.start_as_current_span(name, attributes=span_attributes) as span:
            yield span
    else:
        yield _NoOpSpan()


def _reset_for_testing() -> None:
    """Reset tracing state. Only for use in test fixtures."""
    global _tracer, _initialized
    with _init_lock:
        _tracer = None
        _initialized = False
