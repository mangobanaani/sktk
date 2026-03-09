"""Declarative agent definitions -- load agents from YAML/dict configs.

Enables configuration-driven development where agents are defined
in YAML files rather than code.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from sktk.agent.agent import SKTKAgent
from sktk.agent.capabilities import Capability
from sktk.agent.filters import (
    ContentSafetyFilter,
    PIIFilter,
    PromptInjectionFilter,
    TokenBudgetFilter,
)
from sktk.session.session import Session

# Built-in filter registry
_FILTER_REGISTRY: dict[str, type] = {
    "content_safety": ContentSafetyFilter,
    "pii": PIIFilter,
    "prompt_injection": PromptInjectionFilter,
    "token_budget": TokenBudgetFilter,
}
_filter_lock = threading.Lock()


def register_filter(name: str, cls: type) -> None:
    """Register a custom filter class for use in agent definitions."""
    with _filter_lock:
        _FILTER_REGISTRY[name] = cls


def load_agent_from_dict(config: dict[str, Any]) -> SKTKAgent:
    """Create an SKTKAgent from a configuration dictionary.

    Expected structure:
        {
            "name": "analyst",
            "instructions": "You analyze data.",
            "max_iterations": 5,
            "timeout": 30.0,
            "filters": [
                {"type": "prompt_injection"},
                {"type": "pii"},
                {"type": "content_safety", "blocked_patterns": ["badword"]},
                {"type": "token_budget", "max_tokens": 4000},
            ],
            "capabilities": [
                {"name": "analysis", "description": "Analyze data", "tags": ["finance"]},
            ],
        }
    """
    if not isinstance(config, dict):
        raise TypeError(f"Agent config must be a dict, got {type(config).__name__}")
    for required_field in ("name", "instructions"):
        if required_field not in config:
            raise ValueError(f"Agent config missing required field: {required_field!r}")
    if not isinstance(config["name"], str):
        raise TypeError(f"'name' must be a string, got {type(config['name']).__name__}")
    if not isinstance(config["instructions"], str):
        raise TypeError(
            f"'instructions' must be a string, got {type(config['instructions']).__name__}"
        )

    if "tools" in config:
        import warnings

        warnings.warn(
            "The 'tools' field in agent config is not supported by load_agent_from_dict "
            "and will be ignored. Register tools programmatically after loading.",
            UserWarning,
            stacklevel=2,
        )

    name = config["name"]
    instructions = config["instructions"]

    filters = []
    for f_config in config.get("filters", []):
        f_type = f_config["type"]
        with _filter_lock:
            if f_type not in _FILTER_REGISTRY:
                raise ValueError(
                    f"Unknown filter type: {f_type!r}. Available: {list(_FILTER_REGISTRY)}"
                )
            f_cls = _FILTER_REGISTRY[f_type]
        f_kwargs = {k: v for k, v in f_config.items() if k != "type"}
        filters.append(f_cls(**f_kwargs))

    capabilities = []
    for c_config in config.get("capabilities", []):
        capabilities.append(
            Capability(
                name=c_config["name"],
                description=c_config.get("description", ""),
                input_types=[],
                output_types=[],
                tags=c_config.get("tags", []),
            )
        )

    session = None
    if "session_id" in config:
        session = Session(id=config["session_id"])

    return SKTKAgent(
        name=name,
        instructions=instructions,
        session=session,
        capabilities=capabilities,
        filters=filters,
        max_iterations=config.get("max_iterations", 10),
        timeout=config.get("timeout", 60.0),
    )


def load_agent_from_yaml(path: str | Path) -> SKTKAgent:
    """Load an SKTKAgent from a YAML file.

    Requires PyYAML (pip install pyyaml).
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "YAML agent definitions require PyYAML. Install with: pip install pyyaml"
        ) from e

    path = Path(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    return load_agent_from_dict(config)


def load_agent_from_json(path: str | Path) -> SKTKAgent:
    """Load an SKTKAgent from a JSON file."""
    path = Path(path)
    with path.open() as f:
        config = json.load(f)
    return load_agent_from_dict(config)
