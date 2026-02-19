"""Event stream + structured logging integration.

Demonstrates forwarding typed runtime events into an EventStream sink that logs
JSON records enriched with execution context.

Usage:
    python examples/concepts/observability/event_stream_logging.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from _provider import get_provider

from sktk import BroadcastStrategy, CompletionEvent, MessageEvent, Session, SKTKAgent, SKTKTeam
from sktk.core.context import context_scope
from sktk.observability import EventSink, EventStream, configure_structured_logging, get_logger


class StructuredLoggingSink(EventSink):
    """Send every event to a structured context-aware logger."""

    def __init__(self, logger_name: str = "sktk.events") -> None:
        self._logger = get_logger(logger_name)

    async def send(self, event: Any) -> None:
        payload = asdict(event) if hasattr(event, "__dataclass_fields__") else {"value": str(event)}
        self._logger.info(
            "event_emitted",
            event_kind=getattr(event, "kind", type(event).__name__),
            event_payload=payload,
        )


async def main() -> None:
    provider = get_provider()
    configure_structured_logging(logging.INFO)

    stream = EventStream(sinks=[StructuredLoggingSink()])

    researcher = SKTKAgent(
        name="researcher",
        instructions="You are a launch risk researcher. Identify key risks. Be concise.",
        service=provider,
        timeout=30.0,
    )
    analyst = SKTKAgent(
        name="analyst",
        instructions="You are a risk analyst. Prioritize the risks identified. Be concise.",
        service=provider,
        timeout=30.0,
    )

    team = SKTKTeam(
        agents=[researcher, analyst],
        strategy=BroadcastStrategy(),
        session=Session(id="event-stream-demo"),
    )

    async with context_scope(tenant_id="demo", user_id="u-1", session_id="event-stream-demo"):
        async for event in team.stream("Assess launch risks"):
            await stream.emit(event)

            if isinstance(event, MessageEvent):
                print(f"message: {event.agent} -> {event.content}")
            elif isinstance(event, CompletionEvent):
                print(
                    f"completion: rounds={event.total_rounds}, duration={event.duration_seconds:.3f}s"
                )

    print(f"Recorded events: {len(stream)}")


if __name__ == "__main__":
    asyncio.run(main())
