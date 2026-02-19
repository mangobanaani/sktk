"""Lifecycle hooks and middleware.

Hooks fire at key points in every agent invocation -- start, complete,
error -- without modifying agent logic.  Middleware wraps invoke() to
add cross-cutting concerns (timing, logging, caching).

Usage:
    python examples/getting_started/04_lifecycle_hooks.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _provider import get_provider

from sktk import SKTKAgent
from sktk.agent.hooks import LifecycleHooks
from sktk.agent.middleware import MiddlewareStack

# -- Hooks: observe without modifying --


async def log_start(agent_name: str, input_text: str) -> None:
    print(f"  [hook:start]    {agent_name} <- {input_text!r}")


async def log_complete(agent_name: str, input_text: str, output: object) -> None:
    print(f"  [hook:complete] {agent_name} -> {str(output)[:60]!r}")


async def log_error(agent_name: str, input_text: str, error: Exception) -> None:
    print(f"  [hook:error]    {agent_name} raised {error!r}")


# -- Middleware: wrap invoke() --


async def timing_mw(agent_name: str, message: str, next_fn: object) -> object:
    """Measure wall-clock time for each invocation."""
    start = time.monotonic()
    result = await next_fn(message)  # type: ignore[operator]
    elapsed = (time.monotonic() - start) * 1000
    print(f"  [mw:timing]     {agent_name} took {elapsed:.1f}ms")
    return result


async def uppercase_mw(agent_name: str, message: str, next_fn: object) -> object:
    """Post-process the response to uppercase (silly but illustrative)."""
    result = await next_fn(message)  # type: ignore[operator]
    return str(result).upper()


async def main() -> None:
    provider = get_provider()

    # 1) Hooks -- lightweight observers
    print("=== Lifecycle Hooks ===")
    hooks = LifecycleHooks(
        on_start=[log_start],
        on_complete=[log_complete],
        on_error=[log_error],
    )
    agent = SKTKAgent(
        name="hooked-agent",
        instructions="You are a helpful assistant. Be concise.",
        service=provider,
        timeout=30.0,
        hooks=hooks,
    )
    await agent.invoke("What is the capital of Finland?")

    # 2) Middleware -- wraps invoke with a chain
    print("\n=== Middleware Stack ===")
    stack = MiddlewareStack()
    stack.add(timing_mw)
    stack.add(uppercase_mw)

    agent2 = SKTKAgent(
        name="mw-agent",
        instructions="You are a helpful assistant. Be concise.",
        service=provider,
        timeout=30.0,
    )

    # Wrap the agent's invoke so every call goes through the stack
    wrapped = stack.wrap(agent2.invoke, agent_name=agent2.name)
    result = await wrapped("When was Rust released?")
    print(f"  final result: {result!r}")


if __name__ == "__main__":
    asyncio.run(main())
