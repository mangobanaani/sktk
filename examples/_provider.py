"""Shared provider setup for examples."""

from pathlib import Path

from sktk.agent.providers import AnthropicClaudeProvider
from sktk.core.secrets import FileSecretsProvider

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256


def get_provider() -> AnthropicClaudeProvider:
    import anthropic

    root = Path(__file__).resolve().parent.parent
    secrets = FileSecretsProvider(root / ".env")
    key = secrets.require("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=key)
    return AnthropicClaudeProvider(client=client, model=MODEL, max_tokens=MAX_TOKENS)
