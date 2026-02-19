"""OpenTelemetry metrics for SKTK operations."""

from __future__ import annotations

from typing import Any, Callable

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

_meter = None
_initialized = False


def instrument_metrics(exporter: Any = None) -> None:
    """Initialize OpenTelemetry metrics for SKTK. Idempotent."""
    global _meter, _initialized
    if _initialized:
        return
    if not _HAS_OTEL:
        _initialized = True
        return
    if exporter:
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
    _meter = metrics.get_meter("skat", "0.1.0")
    _initialized = True


def _get_meter():
    if _meter is None:
        instrument_metrics()
    return _meter


def make_metrics_hook(prefix: str = "sktk.checkpoint") -> Callable[[str, dict[str, Any]], None]:
    """Return a metrics hook for CheckpointStore operations."""
    if not _HAS_OTEL:

        def _noop(event: str, payload: dict[str, Any]) -> None:
            return None

        return _noop

    meter = _get_meter()
    counter = meter.create_counter(f"{prefix}.ops")
    duration = meter.create_histogram(f"{prefix}.duration_ms")

    def _hook(event: str, payload: dict[str, Any]) -> None:
        attrs = {k: str(v) for k, v in payload.items() if k != "duration_ms"}
        counter.add(1, attrs)
        if "duration_ms" in payload:
            try:
                duration.record(float(payload["duration_ms"]), attrs)
            except Exception:
                return None

    return _hook


def _reset_for_testing() -> None:
    global _meter, _initialized
    _meter = None
    _initialized = False
