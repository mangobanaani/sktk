"""Conversation summarization for token management.

Provides strategies to compress long conversation histories
while preserving critical context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result of summarizing a conversation."""

    messages: list[dict[str, Any]]
    original_count: int
    summarized_count: int
    summary_text: str


class WindowSummarizer:
    """Keep only the most recent N messages, with an optional system summary.

    The simplest and most deterministic approach. Works well when
    recent context is more important than historical.

    Usage:
        summarizer = WindowSummarizer(window_size=20)
        result = summarizer.summarize(messages)
    """

    def __init__(self, window_size: int = 20, keep_system: bool = True) -> None:
        self._window_size = window_size
        self._keep_system = keep_system

    def summarize(self, messages: list[dict[str, Any]]) -> SummaryResult:
        system_msgs = []
        non_system = []
        for m in messages:
            if self._keep_system and m.get("role") == "system":
                system_msgs.append(m)
            else:
                non_system.append(m)

        if len(non_system) <= self._window_size:
            return SummaryResult(
                messages=messages,
                original_count=len(messages),
                summarized_count=len(messages),
                summary_text="",
            )

        dropped_count = max(0, len(non_system) - self._window_size)
        kept = non_system[-self._window_size :]

        summary = f"[{dropped_count} earlier messages summarized]"
        summary_msg = {"role": "system", "content": summary}

        result_messages = system_msgs + [summary_msg] + kept
        return SummaryResult(
            messages=result_messages,
            original_count=len(messages),
            summarized_count=len(result_messages),
            summary_text=summary,
        )


class TokenBudgetSummarizer:
    """Keep messages within a token budget using word-count estimation.

    Estimates tokens as words * 1.3 (rough average for English text).
    Trims oldest non-system messages first.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        tokens_per_word: float = 1.3,
        keep_system: bool = True,
    ) -> None:
        self._max_tokens = max_tokens
        self._tokens_per_word = tokens_per_word
        self._keep_system = keep_system

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * self._tokens_per_word)

    def summarize(self, messages: list[dict[str, Any]]) -> SummaryResult:
        total_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in messages)

        if total_tokens <= self._max_tokens:
            return SummaryResult(
                messages=messages,
                original_count=len(messages),
                summarized_count=len(messages),
                summary_text="",
            )

        system_msgs = []
        non_system = []
        for m in messages:
            if self._keep_system and m.get("role") == "system":
                system_msgs.append(m)
            else:
                non_system.append(m)

        system_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in system_msgs)
        budget = max(0, self._max_tokens - system_tokens - 50)  # reserve for summary msg

        kept: list[dict[str, Any]] = []
        used = 0
        for m in reversed(non_system):
            tokens = self._estimate_tokens(m.get("content", ""))
            if used + tokens > budget:
                break
            kept.append(m)
            used += tokens

        if not kept and non_system:
            kept.append(non_system[-1])
            logger.warning("Token budget too small; keeping only the most recent message")

        kept.reverse()

        dropped = len(non_system) - len(kept)
        summary = f"[{dropped} earlier messages trimmed to fit token budget]"
        summary_msg = {"role": "system", "content": summary}

        result_messages = system_msgs + [summary_msg] + kept
        return SummaryResult(
            messages=result_messages,
            original_count=len(messages),
            summarized_count=len(result_messages),
            summary_text=summary,
        )
