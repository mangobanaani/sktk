# tests/unit/core/test_config.py
import importlib
import sys
import warnings

import pytest

from sktk.core.config import LoggingConfig, ModelConfig, RetryConfig, SKTKConfig


def test_default_config():
    config = SKTKConfig()
    assert config.model.provider == "openai"
    assert config.retry.max_retries == 3
    assert config.logging.level == "INFO"
    assert config.default_timeout == 60.0


def test_from_dict():
    config = SKTKConfig.from_dict(
        {
            "model": {"provider": "anthropic", "model_name": "claude-3"},
            "retry": {"max_retries": 5},
            "default_timeout": 30.0,
        }
    )
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "claude-3"
    assert config.retry.max_retries == 5
    assert config.default_timeout == 30.0


def test_from_dict_empty():
    config = SKTKConfig.from_dict({})
    assert config.model.provider == "openai"


def test_from_env(monkeypatch):
    monkeypatch.setenv("SKTK_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("SKTK_MODEL_NAME", "claude-3")
    monkeypatch.setenv("SKTK_RETRY_MAX_RETRIES", "5")
    monkeypatch.setenv("SKTK_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SKTK_DEFAULT_TIMEOUT", "30.0")

    config = SKTKConfig.from_env()
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "claude-3"
    assert config.retry.max_retries == 5
    assert config.logging.level == "DEBUG"
    assert config.default_timeout == 30.0


def test_from_env_defaults():
    config = SKTKConfig.from_env(prefix="NONEXISTENT_PREFIX_")
    assert config.model.provider == "openai"
    assert config.retry.max_retries == 3


def test_model_config():
    m = ModelConfig(provider="test", model_name="test-model", temperature=0.5)
    assert m.provider == "test"
    assert m.temperature == 0.5


def test_retry_config():
    r = RetryConfig(max_retries=10, base_delay=0.5)
    assert r.max_retries == 10
    assert r.base_delay == 0.5


def test_logging_config():
    cfg = LoggingConfig(level="DEBUG", structured=False)
    assert cfg.level == "DEBUG"
    assert cfg.structured is False


def test_from_env_structured_false(monkeypatch):
    monkeypatch.setenv("SKTK_LOG_STRUCTURED", "false")
    config = SKTKConfig.from_env()
    assert config.logging.structured is False


def test_from_yaml_success(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  provider: anthropic",
                "  model_name: claude-3",
                "retry:",
                "  max_retries: 7",
                "default_timeout: 42.0",
            ]
        ),
        encoding="utf-8",
    )

    config = SKTKConfig.from_yaml(path)

    assert config.model.provider == "anthropic"
    assert config.model.model_name == "claude-3"
    assert config.retry.max_retries == 7
    assert config.default_timeout == 42.0


def test_from_yaml_missing_yaml_dependency_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(ImportError, match="YAML config requires PyYAML"):
        SKTKConfig.from_yaml("unused.yaml")


def test_from_yaml_none_payload_uses_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    config = SKTKConfig.from_yaml(path)

    assert config.model.provider == "openai"
    assert config.retry.max_retries == 3
    assert config.default_timeout == 60.0


def test_model_config_does_not_emit_pydantic_v1_validator_warnings():
    pydantic = pytest.importorskip("pydantic")
    warning_class = getattr(pydantic.warnings, "PydanticDeprecatedSince20", None)
    if warning_class is None:
        pytest.skip("PydanticDeprecatedSince20 not available")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", warning_class)
        importlib.reload(sys.modules["sktk.core.config"])
        ModelConfig(provider="test", model_name="test-model", temperature=0.5)
    assert not any(isinstance(w.message, warning_class) for w in captured)
