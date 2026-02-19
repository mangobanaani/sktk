# tests/unit/agent/test_loader.py
import builtins
import json
import tempfile

import pytest

from sktk.agent.loader import (
    _FILTER_REGISTRY,
    load_agent_from_dict,
    load_agent_from_json,
    load_agent_from_yaml,
    register_filter,
)


def test_load_minimal_agent():
    config = {"name": "test", "instructions": "Do things."}
    agent = load_agent_from_dict(config)
    assert agent.name == "test"
    assert agent.instructions == "Do things."
    assert agent.filters == []
    assert agent.capabilities == []


def test_load_with_filters():
    config = {
        "name": "safe",
        "instructions": "Be safe.",
        "filters": [
            {"type": "content_safety", "blocked_patterns": ["bad"]},
            {"type": "pii"},
        ],
    }
    agent = load_agent_from_dict(config)
    assert len(agent.filters) == 2


def test_load_with_capabilities():
    config = {
        "name": "skilled",
        "instructions": "I can do things.",
        "capabilities": [
            {"name": "search", "description": "Search the web", "tags": ["info"]},
            {"name": "calc"},
        ],
    }
    agent = load_agent_from_dict(config)
    assert len(agent.capabilities) == 2
    assert agent.capabilities[0].name == "search"
    assert agent.capabilities[0].tags == ["info"]
    assert agent.capabilities[1].description == ""


def test_load_with_session_id():
    config = {"name": "sessioned", "instructions": "Hi.", "session_id": "s123"}
    agent = load_agent_from_dict(config)
    assert agent.session is not None
    assert agent.session.id == "s123"


def test_load_with_max_iterations_and_timeout():
    config = {
        "name": "tuned",
        "instructions": "Tune.",
        "max_iterations": 3,
        "timeout": 15.0,
    }
    agent = load_agent_from_dict(config)
    assert agent.max_iterations == 3
    assert agent.timeout == 15.0


def test_load_unknown_filter_raises():
    config = {
        "name": "bad",
        "instructions": "Hmm.",
        "filters": [{"type": "nonexistent_filter"}],
    }
    with pytest.raises(ValueError, match="Unknown filter type"):
        load_agent_from_dict(config)


def test_register_custom_filter():
    class MyFilter:
        pass

    register_filter("my_custom", MyFilter)
    assert "my_custom" in _FILTER_REGISTRY
    # Cleanup
    del _FILTER_REGISTRY["my_custom"]


def test_load_from_json():
    config = {"name": "json_agent", "instructions": "From JSON."}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        agent = load_agent_from_json(f.name)
    assert agent.name == "json_agent"


def test_load_from_yaml():
    pytest.importorskip("yaml")
    content = "name: yaml_agent\ninstructions: From YAML.\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        agent = load_agent_from_yaml(f.name)
    assert agent.name == "yaml_agent"
    assert agent.instructions == "From YAML."


def test_load_from_yaml_without_pyyaml_raises(monkeypatch, tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("name: yaml_agent\ninstructions: From YAML.\n")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="require PyYAML"):
        load_agent_from_yaml(path)


def test_load_with_token_budget_filter():
    config = {
        "name": "budgeted",
        "instructions": "Stay within budget.",
        "filters": [{"type": "token_budget", "max_tokens": 2000}],
    }
    agent = load_agent_from_dict(config)
    assert len(agent.filters) == 1


def test_load_with_prompt_injection_filter():
    config = {
        "name": "protected",
        "instructions": "Stay safe.",
        "filters": [{"type": "prompt_injection"}],
    }
    agent = load_agent_from_dict(config)
    assert len(agent.filters) == 1
