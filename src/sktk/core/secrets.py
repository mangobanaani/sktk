"""Secrets management with pluggable providers.

Abstracts API key and credential management behind a protocol,
supporting environment variables, files, and vault services.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SecretsProvider(Protocol):
    """Protocol for secrets providers."""

    def get(self, key: str) -> str | None: ...
    def require(self, key: str) -> str: ...


class EnvSecretsProvider:
    """Load secrets from environment variables.

    Usage:
        secrets = EnvSecretsProvider(prefix="SKTK_")
        api_key = secrets.require("OPENAI_API_KEY")
        # Looks up SKTK_OPENAI_API_KEY
    """

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def get(self, key: str) -> str | None:
        return os.environ.get(f"{self._prefix}{key}")

    def require(self, key: str) -> str:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Required secret '{self._prefix}{key}' not found in environment")
        return val


class FileSecretsProvider:
    """Load secrets from a dotenv-style file.

    Usage:
        secrets = FileSecretsProvider(".env")
        api_key = secrets.get("OPENAI_API_KEY")
    """

    def __init__(self, path: str | Path) -> None:
        self._secrets: dict[str, str] = {}
        path = Path(path)
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning("Skipping unparseable line in %s: %r", path, line)
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[7:].strip()
                if not key:
                    logger.warning("Skipping line with empty key in %s: %r", path, line)
                    continue
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                self._secrets[key] = val

    def get(self, key: str) -> str | None:
        return self._secrets.get(key)

    def require(self, key: str) -> str:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Required secret '{key}' not found in secrets file")
        return val


class ChainedSecretsProvider:
    """Try multiple providers in order.

    Usage:
        secrets = ChainedSecretsProvider([
            FileSecretsProvider(".env.local"),
            EnvSecretsProvider(),
        ])
    """

    def __init__(self, providers: list[SecretsProvider]) -> None:
        self._providers = providers

    def get(self, key: str) -> str | None:
        for provider in self._providers:
            val = provider.get(key)
            if val is not None:
                return val
        return None

    def require(self, key: str) -> str:
        val = self.get(key)
        if val is None:
            sources = [type(p).__name__ for p in self._providers]
            raise KeyError(f"Required secret '{key}' not found in any provider: {sources}")
        return val
