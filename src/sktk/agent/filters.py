"""Guardrail filter pipeline for agents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from sktk.core.types import Allow, Deny, FilterResult, Modify

FilterStage = Literal["input", "output", "function_call"]


@dataclass
class FilterContext:
    """Context passed through filter pipeline."""

    content: str
    stage: FilterStage
    agent_name: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentFilter(Protocol):
    """Protocol for guardrail filters."""

    async def on_input(self, context: FilterContext) -> FilterResult: ...
    async def on_function_call(self, context: FilterContext) -> FilterResult: ...
    async def on_output(self, context: FilterContext) -> FilterResult: ...


class FilterAdapter:
    """Base class with default Allow() for all filter hooks.

    Subclass and override only the methods you need.  The
    :class:`AgentFilter` protocol is kept as-is for duck-typing;
    this adapter simply removes the boilerplate of returning
    ``Allow()`` from unused hooks.
    """

    async def on_input(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return Allow()

    async def on_output(self, context: FilterContext) -> FilterResult:
        return Allow()


class ContentSafetyFilter(FilterAdapter):
    """Block content matching configurable regex patterns."""

    def __init__(self, blocked_patterns: list[str]) -> None:
        self._patterns = [re.compile(p) for p in blocked_patterns]

    async def on_input(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_output(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    def _check(self, content: str) -> FilterResult:
        """Return Deny if content matches any blocked pattern."""
        for pattern in self._patterns:
            if pattern.search(content):
                return Deny(reason=f"Content matched blocked pattern: {pattern.pattern}")
        return Allow()


class PIIFilter(FilterAdapter):
    """Detect common PII patterns (email, phone, SSN).

    Pass a custom *patterns* list to override the defaults.  Each entry
    is a ``(regex, description)`` tuple.
    """

    DEFAULT_PATTERNS: list[tuple[str, str]] = [
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email address"),
        # SSN: require separators to avoid matching arbitrary 9-digit numbers
        (r"\b\d{3}[-.]\d{2}[-.]\d{4}\b", "SSN-like number"),
        # Phone: require separators or parenthesized area code
        (r"(?:\b\d{3}[-.]\d{3}[-.]\d{4}\b|\(\d{3}\)\s*\d{3}[-.]\d{4})", "phone number"),
        # Credit card numbers (Visa, MasterCard, Amex, Discover)
        (
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "credit card number",
        ),
        # IPv4 addresses (but not version numbers like 1.2.3)
        (
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
            "IP address",
        ),
        # IBAN (international bank account, minimum 15 characters)
        (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b", "IBAN"),
        # Passport number (exactly 2 uppercase letters + 7 digits, not
        # followed by more alphanumerics to avoid matching IBAN prefixes)
        (r"\b[A-Z]{2}\d{7}(?![A-Z0-9])", "passport number"),
    ]

    def __init__(self, patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = patterns if patterns is not None else self.DEFAULT_PATTERNS
        self._compiled = [(re.compile(p), desc) for p, desc in self._patterns]

    async def on_input(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_output(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    def _check(self, content: str) -> FilterResult:
        """Return Deny if content contains any PII pattern."""
        for pattern, desc in self._compiled:
            if pattern.search(content):
                return Deny(reason=f"PII detected: {desc}")
        return Allow()


class TokenBudgetFilter(FilterAdapter):
    """Deny requests exceeding a token budget."""

    def __init__(self, max_tokens: int) -> None:
        self._max = max_tokens

    async def on_input(self, context: FilterContext) -> FilterResult:
        if context.token_count is not None and context.token_count > self._max:
            return Deny(reason=f"Token count {context.token_count} exceeds budget {self._max}")
        return Allow()


class PromptInjectionFilter(FilterAdapter):
    """Detect common prompt injection and jailbreak patterns.

    Catches attempts to override system instructions, extract system
    prompts, or bypass guardrails through known injection techniques.

    Note: This filter operates on ASCII-normalized text after NFKD
    decomposition and homoglyph transliteration. Injection patterns written
    entirely in non-Latin scripts (e.g., CJK, Arabic) that do not contain
    recognizable ASCII injection keywords will not be detected. For
    multilingual deployments, consider supplementing this filter with an
    LLM-based content classifier.
    """

    _ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060]")
    _WHITESPACE_RE = re.compile(r"\s+")

    # Common Cyrillic -> Latin homoglyph mapping
    _HOMOGLYPHS = str.maketrans(
        {
            "\u0430": "a",
            "\u0435": "e",
            "\u043e": "o",
            "\u0440": "p",
            "\u0441": "c",
            "\u0443": "y",
            "\u0445": "x",
            "\u0456": "i",
            "\u0458": "j",
            "\u04bb": "h",
            "\u0432": "b",
            "\u043c": "m",
            "\u043d": "n",
            "\u0442": "t",
            "\u0410": "A",
            "\u0412": "B",
            "\u0415": "E",
            "\u041a": "K",
            "\u041c": "M",
            "\u041d": "H",
            "\u041e": "O",
            "\u0420": "P",
            "\u0421": "C",
            "\u0422": "T",
            "\u0423": "Y",
            "\u0425": "X",
        }
    )

    _INJECTION_PATTERNS = [
        (
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            "instruction override",
        ),
        (
            r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            "instruction override",
        ),
        (
            r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|prompts|rules)",
            "instruction override",
        ),
        (r"you\s+are\s+now\s+(a|an|the)\s+", "role reassignment"),
        (r"pretend\s+(you\s+are|to\s+be)\s+", "role reassignment"),
        (r"act\s+as\s+(if\s+you\s+are|a|an)\s+", "role reassignment"),
        (
            r"(reveal|show|print|output|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions)",
            "system prompt extraction",
        ),
        (
            r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)",
            "system prompt extraction",
        ),
        (r"\[system\]", "injected system tag"),
        (r"<\|?system\|?>", "injected system tag"),
        (r"###\s*(system|instruction)", "injected system tag"),
        (r"do\s+anything\s+now", "DAN jailbreak"),
        (r"jailbreak", "explicit jailbreak"),
        (r"bypass\s+(all\s+)?(safety|filter|restriction|guardrail)", "guardrail bypass"),
    ]

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self._compiled = [(re.compile(p), desc) for p, desc in self._INJECTION_PATTERNS]
        if extra_patterns:
            self._compiled.extend((re.compile(p), "custom pattern") for p in extra_patterns)

    async def on_input(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_function_call(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    async def on_output(self, context: FilterContext) -> FilterResult:
        return self._check(context.content)

    def _check(self, content: str) -> FilterResult:
        """Return Deny if content matches any injection pattern."""
        normalized = self._normalize(content)
        for pattern, desc in self._compiled:
            if pattern.search(normalized):
                return Deny(reason=f"Prompt injection detected: {desc}")
        return Allow()

    def _normalize(self, content: str) -> str:
        """Normalize Unicode obfuscation, lowercase, strip zero-width chars, collapse whitespace."""
        # NFKD decomposition normalizes full-width chars, compatibility forms
        content = unicodedata.normalize("NFKD", content)
        # Transliterate known Cyrillic/Greek homoglyphs to Latin equivalents
        content = content.translate(self._HOMOGLYPHS)
        # Strip zero-width characters before ASCII encode
        content = self._ZERO_WIDTH_RE.sub("", content)
        content = content.encode("ascii", "ignore").decode("ascii")
        content = content.lower()
        content = self._WHITESPACE_RE.sub(" ", content)
        return content


async def run_chunk_filters(
    filters: list[AgentFilter],
    context: FilterContext,
) -> FilterResult:
    """Run on_output_chunk for filters that implement it. Others are skipped."""
    for f in filters:
        handler = getattr(f, "on_output_chunk", None)
        if handler is None:
            continue
        result = await handler(context)
        if isinstance(result, Deny):
            return result
    return Allow()


async def run_filter_pipeline(
    filters: list[AgentFilter],
    context: FilterContext,
) -> FilterResult:
    """Execute filters in order. Deny short-circuits. Modify updates content."""
    stage = context.stage
    last_modified: str | None = None
    for f in filters:
        # Dispatch to the correct handler based on pipeline stage
        if stage == "input":
            handler = f.on_input
        elif stage == "output":
            handler = f.on_output
        elif stage == "function_call":
            handler = f.on_function_call
        else:
            raise ValueError(f"Unknown filter stage: {stage!r}")
        result = await handler(context)
        if isinstance(result, Deny):
            return result
        if isinstance(result, Modify):
            last_modified = result.content
            context = FilterContext(
                content=result.content,
                stage=context.stage,
                agent_name=context.agent_name,
                token_count=context.token_count,
                metadata=context.metadata.copy(),
            )
    if last_modified is not None:
        return Modify(content=last_modified)
    return Allow()
