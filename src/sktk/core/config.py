"""Configuration management for SKTK applications.

Loads settings from environment variables, YAML files, or dicts.
Follows 12-factor app principles with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    BaseModel = object  # type: ignore[assignment, misc]  # Fallback if pydantic not available
    Field = field  # type: ignore[assignment]
    ValidationError = ValueError  # type: ignore[assignment, misc]

    def field_validator(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _decorator(func: Any) -> Any:
            return func

        return _decorator


@dataclass
class ModelConfig:
    """LLM model configuration."""

    provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str = field(default="", repr=False)


class _ModelConfigPydantic(BaseModel):
    """Pydantic validation for ModelConfig."""

    provider: str = Field(default="openai", description="LLM provider name")
    model_name: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, ge=1, le=8192, description="Maximum tokens to generate")
    api_key: str = Field(default="", description="API key for the provider")

    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1 or v > 8192:
            raise ValueError("max_tokens must be between 1 and 8192")
        return v


@dataclass
class RetryConfig:
    """Retry policy configuration."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: str = "exponential_jitter"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    structured: bool = True


@dataclass
class SKTKConfig:
    """Top-level SKTK configuration.

    Usage:
        # From environment
        config = SKTKConfig.from_env()

        # From dict
        config = SKTKConfig.from_dict({"model": {"provider": "anthropic"}})

        # From YAML
        config = SKTKConfig.from_yaml("config.yaml")
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    default_timeout: float = 60.0
    max_iterations: int = 10

    def validate(self) -> None:
        """Validate configuration using Pydantic models."""
        try:
            _ModelConfigPydantic(**self.model.__dict__)
            # Add validation for other configs as needed
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}") from e

    @classmethod
    def from_env(cls, prefix: str = "SKTK_") -> SKTKConfig:
        """Load configuration from environment variables.

        Environment variables use the pattern: SKTK_SECTION_KEY
        e.g., SKTK_MODEL_PROVIDER, SKTK_RETRY_MAX_RETRIES
        """

        def env(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}{key}", default)

        def _parse_float(var_name: str, raw: str) -> float:
            try:
                return float(raw)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {prefix}{var_name}: expected float, got {raw!r}"
                ) from None

        def _parse_int(var_name: str, raw: str) -> int:
            try:
                return int(raw)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {prefix}{var_name}: expected integer, got {raw!r}"
                ) from None

        model = ModelConfig(
            provider=env("MODEL_PROVIDER", "openai"),
            model_name=env("MODEL_NAME", "gpt-4"),
            temperature=_parse_float("MODEL_TEMPERATURE", env("MODEL_TEMPERATURE", "0.7")),
            max_tokens=_parse_int("MODEL_MAX_TOKENS", env("MODEL_MAX_TOKENS", "4096")),
            api_key=env("MODEL_API_KEY"),
        )
        retry = RetryConfig(
            max_retries=_parse_int("RETRY_MAX_RETRIES", env("RETRY_MAX_RETRIES", "3")),
            base_delay=_parse_float("RETRY_BASE_DELAY", env("RETRY_BASE_DELAY", "1.0")),
            max_delay=_parse_float("RETRY_MAX_DELAY", env("RETRY_MAX_DELAY", "60.0")),
            backoff=env("RETRY_BACKOFF", "exponential_jitter"),
        )
        logging_cfg = LoggingConfig(
            level=env("LOG_LEVEL", "INFO"),
            structured=env("LOG_STRUCTURED", "true").lower() in ("true", "1", "yes"),
        )
        return cls(
            model=model,
            retry=retry,
            logging=logging_cfg,
            default_timeout=_parse_float("DEFAULT_TIMEOUT", env("DEFAULT_TIMEOUT", "60.0")),
            max_iterations=_parse_int("MAX_ITERATIONS", env("MAX_ITERATIONS", "10")),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SKTKConfig:
        """Load configuration from a dictionary."""
        model_data = data.get("model", {})
        retry_data = data.get("retry", {})
        logging_data = data.get("logging", {})

        return cls(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            retry=RetryConfig(**retry_data) if retry_data else RetryConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
            default_timeout=data.get("default_timeout", 60.0),
            max_iterations=data.get("max_iterations", 10),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SKTKConfig:
        """Load configuration from a YAML file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("YAML config requires PyYAML: pip install pyyaml") from e

        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
