# tests/unit/core/test_secrets.py
import tempfile

import pytest

from sktk.core.secrets import ChainedSecretsProvider, EnvSecretsProvider, FileSecretsProvider


def test_env_provider_get(monkeypatch):
    monkeypatch.setenv("MY_KEY", "my_value")
    provider = EnvSecretsProvider()
    assert provider.get("MY_KEY") == "my_value"
    assert provider.get("NONEXISTENT") is None


def test_env_provider_with_prefix(monkeypatch):
    monkeypatch.setenv("SKTK_API_KEY", "secret123")
    provider = EnvSecretsProvider(prefix="SKTK_")
    assert provider.get("API_KEY") == "secret123"


def test_env_provider_require(monkeypatch):
    monkeypatch.setenv("MY_KEY", "val")
    provider = EnvSecretsProvider()
    assert provider.require("MY_KEY") == "val"


def test_env_provider_require_missing():
    provider = EnvSecretsProvider(prefix="NONEXISTENT_PREFIX_")
    with pytest.raises(KeyError, match="not found"):
        provider.require("ANYTHING")


def test_file_provider():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("API_KEY=secret123\n")
        f.write("# comment\n")
        f.write("DB_HOST='localhost'\n")
        f.write('TOKEN="abc"\n')
        f.flush()
        provider = FileSecretsProvider(f.name)
    assert provider.get("API_KEY") == "secret123"
    assert provider.get("DB_HOST") == "localhost"
    assert provider.get("TOKEN") == "abc"
    assert provider.get("MISSING") is None


def test_file_provider_require_returns_value():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("API_KEY=secret123\n")
        f.flush()
        provider = FileSecretsProvider(f.name)
    assert provider.require("API_KEY") == "secret123"


def test_file_provider_missing_file():
    provider = FileSecretsProvider("/nonexistent/.env")
    assert provider.get("KEY") is None


def test_file_provider_require_missing():
    provider = FileSecretsProvider("/nonexistent/.env")
    with pytest.raises(KeyError, match="not found"):
        provider.require("KEY")


def test_chained_provider(monkeypatch):
    monkeypatch.setenv("FROM_ENV", "env_val")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("FROM_FILE=file_val\n")
        f.write("FROM_ENV=should_not_use\n")
        f.flush()
        file_provider = FileSecretsProvider(f.name)

    env_provider = EnvSecretsProvider()
    chained = ChainedSecretsProvider([file_provider, env_provider])

    # File provider checked first
    assert chained.get("FROM_FILE") == "file_val"
    # File provider has FROM_ENV too, uses it first
    assert chained.get("FROM_ENV") == "should_not_use"


def test_chained_provider_require_missing():
    chained = ChainedSecretsProvider([EnvSecretsProvider(prefix="NONEXISTENT_")])
    with pytest.raises(KeyError, match="not found in any provider"):
        chained.require("KEY")


def test_chained_provider_require_returns_value():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("FROM_FILE=file_val\n")
        f.flush()
        chained = ChainedSecretsProvider(
            [EnvSecretsProvider(prefix="NOPE_"), FileSecretsProvider(f.name)]
        )
    assert chained.require("FROM_FILE") == "file_val"
