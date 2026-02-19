# tests/unit/observability/test_logging.py
import json
import logging
from io import StringIO

import pytest

from sktk.core.context import context_scope
from sktk.observability.logging import ContextLogger, StructuredFormatter, get_logger


def test_structured_formatter():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "timestamp" in parsed


def test_structured_formatter_with_context_fields():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test",
        args=None,
        exc_info=None,
    )
    record.correlation_id = "c1"
    record.session_id = "s1"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["correlation_id"] == "c1"
    assert parsed["session_id"] == "s1"


def test_structured_formatter_with_error():
    formatter = StructuredFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="failed",
        args=None,
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["error"]["type"] == "ValueError"
    assert parsed["error"]["message"] == "boom"


def test_structured_formatter_with_extra_data():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="payload",
        args=None,
        exc_info=None,
    )
    record.extra_data = {"x": 1}
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["data"] == {"x": 1}


def test_get_logger():
    logger = get_logger("sktk.test")
    assert isinstance(logger, ContextLogger)


def test_context_logger_normalize_exc_info_passthrough():
    sentinel = ("a", "b", "c")
    assert ContextLogger._normalize_exc_info(sentinel) is sentinel


@pytest.mark.asyncio
async def test_context_logger_enriches_from_context():
    logger = get_logger("sktk.test_enrich")

    # Set up a handler to capture output
    captured = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    inner_logger = logging.getLogger("sktk.test_enrich")
    handler = CaptureHandler()
    inner_logger.addHandler(handler)
    inner_logger.setLevel(logging.DEBUG)

    try:
        async with context_scope(session_id="s123", correlation_id="c456"):
            logger.info("test message")

        assert len(captured) == 1
        assert captured[0].correlation_id == "c456"
        assert captured[0].session_id == "s123"
    finally:
        inner_logger.removeHandler(handler)


def test_context_logger_explicit_kwargs():
    logger = get_logger("sktk.test_kwargs")

    captured = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    inner_logger = logging.getLogger("sktk.test_kwargs")
    handler = CaptureHandler()
    inner_logger.addHandler(handler)
    inner_logger.setLevel(logging.DEBUG)

    try:
        logger.info("test", agent_name="analyst")
        assert len(captured) == 1
        assert captured[0].agent_name == "analyst"
    finally:
        inner_logger.removeHandler(handler)


def test_context_logger_levels():
    logger = get_logger("sktk.test_levels")

    captured = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.levelname)

    inner_logger = logging.getLogger("sktk.test_levels")
    handler = CaptureHandler()
    inner_logger.addHandler(handler)
    inner_logger.setLevel(logging.DEBUG)

    try:
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        assert captured == ["DEBUG", "INFO", "WARNING", "ERROR"]
    finally:
        inner_logger.removeHandler(handler)


def test_configure_structured_logging_uses_sktk_root():
    # ensure a clean logger state
    root = logging.getLogger("sktk")
    for h in list(root.handlers):
        root.removeHandler(h)

    from sktk.observability.logging import configure_structured_logging

    configure_structured_logging(logging.INFO)

    assert any(isinstance(h.formatter, StructuredFormatter) for h in root.handlers)


def test_configure_structured_logging_adds_structured_handler_when_plain_handler_exists():
    root = logging.getLogger("sktk")
    for h in list(root.handlers):
        root.removeHandler(h)

    root.addHandler(logging.StreamHandler())

    from sktk.observability.logging import configure_structured_logging

    configure_structured_logging(logging.INFO)

    assert any(isinstance(h.formatter, StructuredFormatter) for h in root.handlers)


def test_context_logger_error_with_exc_info_emits_structured_error():
    logger = get_logger("sktk.test_exc")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    inner_logger = logging.getLogger("sktk.test_exc")
    inner_logger.handlers = [handler]
    inner_logger.setLevel(logging.INFO)
    inner_logger.propagate = False

    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            logger.error("failure", exc_info=exc, request_id="r1")
    finally:
        inner_logger.handlers = []
        inner_logger.propagate = True

    payload = stream.getvalue().strip().splitlines()[-1]
    parsed = json.loads(payload)
    assert parsed["error"]["type"] == "RuntimeError"
    assert parsed["error"]["message"] == "boom"
    assert parsed["data"]["request_id"] == "r1"


def test_context_logger_exception_defaults_exc_info():
    logger = get_logger("sktk.test_exception_default")
    captured = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record)

    inner_logger = logging.getLogger("sktk.test_exception_default")
    handler = CaptureHandler()
    inner_logger.addHandler(handler)
    inner_logger.setLevel(logging.DEBUG)

    try:
        logger.exception("exception path")
        assert len(captured) == 1
        assert captured[0].levelname == "ERROR"
        assert captured[0].exc_info is not None
    finally:
        inner_logger.removeHandler(handler)
