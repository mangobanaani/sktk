"""Structured logging with automatic context enrichment.

Provides a logger that automatically includes session_id, agent_name,
and correlation_id from the current ExecutionContext.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON log formatter that outputs structured log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a JSON string with context fields."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include context fields if present
        for key in ("correlation_id", "session_id", "agent_name", "tenant_id", "user_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # Include extra fields
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "stack": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class ContextLogger:
    """Logger that automatically enriches log records with ExecutionContext.

    Usage:
        logger = get_logger("sktk.agent")
        logger.info("Agent invoked", agent_name="analyst")

        # Or with automatic context:
        async with context_scope(session_id="s1"):
            logger.info("Processing request")  # session_id auto-included
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @staticmethod
    def _normalize_exc_info(value: Any) -> Any:
        """Normalize exc_info to a form accepted by stdlib logging.

        Returns True (resolved by stdlib via sys.exc_info()), an
        (exc_type, exc_value, traceback) tuple, or None.
        """
        if value is None:
            return None
        if isinstance(value, BaseException):
            return (type(value), value, value.__traceback__)
        return value

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Emit a log record enriched with the current ExecutionContext."""
        from sktk.core.context import get_context

        exc_info = self._normalize_exc_info(kwargs.pop("exc_info", None))

        extra: dict[str, Any] = {}
        ctx = get_context()
        if ctx is not None:
            extra["correlation_id"] = ctx.correlation_id
            extra["session_id"] = ctx.session_id
            extra["tenant_id"] = ctx.tenant_id
            extra["user_id"] = ctx.user_id

        # Override with explicit kwargs
        for key in ("correlation_id", "session_id", "agent_name", "tenant_id", "user_id"):
            if key in kwargs:
                extra[key] = kwargs.pop(key)

        if kwargs:
            extra["extra_data"] = kwargs

        self._logger.log(level, msg, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log an exception at ERROR level with traceback information."""
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, **kwargs)


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware structured logger."""
    return ContextLogger(name)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure the root skat logger with structured JSON output."""
    logger = logging.getLogger("sktk")
    logger.setLevel(level)
    logger.propagate = False

    has_structured_handler = any(
        isinstance(getattr(handler, "formatter", None), StructuredFormatter)
        for handler in logger.handlers
    )
    if not has_structured_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
