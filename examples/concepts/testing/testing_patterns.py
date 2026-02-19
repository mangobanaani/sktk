"""Testing patterns: mocks, sandbox, prompt regression.

SKTK ships a testing toolkit so you can write fast, deterministic
tests for agents without calling a live LLM.

Usage:
    python examples/concepts/testing/testing_patterns.py
"""

import asyncio

from sktk import Session, SKTKAgent, tool
from sktk.core.events import CompletionEvent, MessageEvent
from sktk.testing import (
    MockKernel,
    PluginSandbox,
    PromptSuite,
    PromptTestCase,
    assert_events_emitted,
    assert_history_contains,
)

# -- 1) MockKernel: script LLM responses --


async def demo_mock_kernel() -> None:
    print("=== MockKernel ===")

    # Queue deterministic responses -- no LLM needed
    mk = MockKernel()
    mk.expect_chat_completion(responses=["Paris", "Berlin"])

    agent = SKTKAgent(name="geo", instructions="Answer geography.", kernel=mk)
    r1 = await agent.invoke("Capital of France?")
    r2 = await agent.invoke("Capital of Germany?")
    print(f"  Response 1: {r1}")
    print(f"  Response 2: {r2}")

    # Verify all expected responses were consumed
    mk.verify()
    print("  All expectations met.")


# -- 2) with_responses: one-liner shorthand --


async def demo_with_responses() -> None:
    print("\n=== with_responses (1-liner) ===")

    # Same as MockKernel ceremony, but in one line
    agent = SKTKAgent.with_responses("bot", ["42", "done"])
    print(f"  {await agent.invoke('meaning of life?')}")
    print(f"  {await agent.invoke('anything else?')}")


# -- 3) Session assertions --


async def demo_session_assertions() -> None:
    print("\n=== Session Assertions ===")

    session = Session(id="test-session")
    agent = SKTKAgent.with_responses(
        "helper",
        ["Python is great!"],
        session=session,
    )
    await agent.invoke("Tell me about Python")

    # Assert the session history contains expected messages
    await assert_history_contains(session, "user", "Python")
    await assert_history_contains(session, "assistant", "great")
    print("  History assertions passed.")


# -- 4) PluginSandbox: test tools in isolation --


async def demo_sandbox() -> None:
    print("\n=== Plugin Sandbox ===")

    @tool(description="Reverse a string")
    async def reverse(text: str) -> str:
        return text[::-1]

    sandbox = PluginSandbox()
    result = await sandbox.run(reverse, text="hello")
    print(f"  Tool: {result.tool_name}")
    print(f"  Output: {result.output}")
    print(f"  Success: {result.success}")

    # Test error handling
    @tool(description="Always fails")
    async def broken(x: int) -> int:
        raise ValueError("oops")

    err_result = await sandbox.run(broken, x=1)
    print(f"  Error tool success: {err_result.success}, error: {err_result.error}")


# -- 5) PromptSuite: regression testing for prompt quality --


async def demo_prompt_suite() -> None:
    print("\n=== Prompt Suite ===")

    agent = SKTKAgent.with_responses(
        "qa",
        [
            "Python was created by Guido van Rossum in 1991.",
            "The capital of France is Paris.",
        ],
    )

    suite = PromptSuite()
    suite.add_case(
        PromptTestCase(
            name="python-origin",
            prompt="Who created Python?",
            expected_contains=["Guido", "1991"],
            expected_not_contains=["Java"],
        )
    )
    suite.add_case(
        PromptTestCase(
            name="france-capital",
            prompt="Capital of France?",
            expected_contains=["Paris"],
        )
    )

    results = await suite.run(agent.invoke)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.case_name}")
        for f in r.failures:
            print(f"         {f}")


# -- 6) Event assertions --


def demo_event_assertions() -> None:
    print("\n=== Event Type Assertions ===")

    # Verify that a stream of events contains expected types in order
    events = [
        MessageEvent.__new__(MessageEvent),
        CompletionEvent.__new__(CompletionEvent),
    ]
    assert_events_emitted(events, [MessageEvent, CompletionEvent])
    print("  Event sequence assertion passed.")


async def main() -> None:
    await demo_mock_kernel()
    await demo_with_responses()
    await demo_session_assertions()
    await demo_sandbox()
    await demo_prompt_suite()
    demo_event_assertions()

    print("\nAll testing patterns demonstrated.")


if __name__ == "__main__":
    asyncio.run(main())
