"""Shared provider setup for examples."""

from pathlib import Path

from sktk.agent.providers import AnthropicClaudeProvider
from sktk.core.secrets import FileSecretsProvider

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256


def get_provider() -> AnthropicClaudeProvider:
    import sys

    import anthropic

    root = Path(__file__).resolve().parent.parent
    secrets = FileSecretsProvider(root / ".env")
    key = secrets.get("ANTHROPIC_API_KEY")
    if not key:
        print("No ANTHROPIC_API_KEY found; skipping example.")
        sys.exit(0)
    client = anthropic.AsyncAnthropic(api_key=key)
    return AnthropicClaudeProvider(client=client, model=MODEL, max_tokens=MAX_TOKENS)
