"""Custom coordination strategy example.

Demonstrates creating a custom strategy and composing
it with built-in strategies using the | operator.

Usage:
    python examples/concepts/multi_agent/custom_strategy.py
"""

import asyncio
from typing import Any

from sktk import (
    ConversationHistory,
    RoundRobinStrategy,
    Session,
    SKTKAgent,
)


class SentimentBasedStrategy:
    """Route to different agents based on sentiment keywords in the task."""

    def __init__(self, positive_agent: str, negative_agent: str) -> None:
        self._positive = positive_agent
        self._negative = negative_agent

    async def next_agent(
        self,
        agents: list[Any],
        history: ConversationHistory | None,
        task: str,
        **kwargs: Any,
    ) -> Any | None:
        task_lower = task.lower()
        positive_words = {"good", "great", "happy", "success", "excellent"}
        negative_words = {"bad", "poor", "fail", "problem", "issue"}

        if any(w in task_lower for w in positive_words):
            target = self._positive
        elif any(w in task_lower for w in negative_words):
            target = self._negative
        else:
            return None

        for agent in agents:
            if agent.name == target:
                return agent
        return None

    def __or__(self, other: Any) -> Any:
        from sktk.team.strategies import ComposedStrategy

        return ComposedStrategy([self, other])


async def main() -> None:
    session = Session(id="custom-strategy-demo")

    optimist = SKTKAgent(name="optimist", instructions="Always see the bright side.")
    pessimist = SKTKAgent(name="pessimist", instructions="Point out risks.")
    neutral = SKTKAgent(name="neutral", instructions="Be balanced.")

    strategy = (
        SentimentBasedStrategy(
            positive_agent="optimist",
            negative_agent="pessimist",
        )
        | RoundRobinStrategy()
    )

    print("Strategy composition: SentimentBased | RoundRobin")
    print("- Positive tasks -> optimist")
    print("- Negative tasks -> pessimist")
    print("- Neutral tasks -> round-robin fallback")

    agents = [optimist, pessimist, neutral]
    tasks = [
        "This is great news about the product launch",
        "There's a problem with the deployment",
        "Summarize the meeting notes",
    ]
    for task in tasks:
        agent = await strategy.next_agent(agents, session.history, task)
        print(f"\n  Task: '{task[:50]}...'")
        print(f"  Routed to: {agent.name if agent else 'None'}")


if __name__ == "__main__":
    asyncio.run(main())
