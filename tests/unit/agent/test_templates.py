# tests/unit/agent/test_templates.py
import builtins
import tempfile
from pathlib import Path

import pytest

from sktk.agent.templates import PromptTemplate, load_prompt, load_prompts


def test_render_basic():
    t = PromptTemplate(name="test", text="Hello {{name}}")
    assert t.render(name="World") == "Hello World"


def test_render_multiple_vars():
    t = PromptTemplate(name="test", text="{{greeting}} {{name}}, you have {{count}} items")
    result = t.render(greeting="Hi", name="Alice", count="5")
    assert result == "Hi Alice, you have 5 items"


def test_variables_property():
    t = PromptTemplate(name="test", text="{{a}} and {{ b }} and {{c}}")
    assert sorted(t.variables) == ["a", "b", "c"]


def test_render_with_defaults():
    t = PromptTemplate(name="test", text="{{greeting}} {{name}}", defaults={"greeting": "Hello"})
    assert t.render(name="Bob") == "Hello Bob"


def test_render_override_defaults():
    t = PromptTemplate(name="test", text="{{greeting}} {{name}}", defaults={"greeting": "Hello"})
    assert t.render(name="Bob", greeting="Hi") == "Hi Bob"


def test_render_missing_variable():
    t = PromptTemplate(name="test", text="{{name}} {{age}}")
    with pytest.raises(ValueError, match="Missing template variables"):
        t.render(name="Alice")


def test_validate_valid():
    t = PromptTemplate(name="test", text="Hello {{name}}")
    assert t.validate() == []


def test_validate_unmatched_braces():
    t = PromptTemplate(name="test", text="Hello {{name}")
    issues = t.validate()
    assert any("Unmatched" in i for i in issues)


def test_load_prompt_simple():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt", delete=False) as f:
        f.write("Hello {{name}}, welcome!")
        f.flush()
        template = load_prompt(f.name)
    assert "name" in template.variables
    assert template.render(name="Alice") == "Hello Alice, welcome!"


def test_load_prompts_directory():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "greet.prompt").write_text("Hello {{name}}")
        (Path(d) / "analyze.prompt").write_text("Analyze {{data}}")
        templates = load_prompts(d)
    assert "greet" in templates
    assert "analyze" in templates


def test_no_variables():
    t = PromptTemplate(name="test", text="No variables here")
    assert t.variables == []
    assert t.render() == "No variables here"


def test_load_prompt_defaults_frontmatter():
    content = """---\nname: greet\ndefaults:\n  punctuation: \"!\"\n---\nHello {{name}}{{punctuation}}\n"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt", delete=False) as f:
        f.write(content)
        f.flush()
        template = load_prompt(f.name)

    assert template.defaults == {"punctuation": "!"}
    assert template.render(name="Alice") == "Hello Alice!"


def test_load_prompt_version_and_hash():
    content = """---\nname: greet\nversion: v1\ndefaults:\n  punctuation: \"!\"\n---\nHello {{name}}{{punctuation}}\n"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".prompt", delete=False) as f:
        f.write(content)
        f.flush()
        template = load_prompt(f.name)

    assert template.version == "v1"
    assert len(template.content_hash) == 64


def test_load_prompt_frontmatter_fallback_parser(monkeypatch, tmp_path):
    content = """---
name: fallback_name
version: v2
audience: analysts
defaults:
  punctuation: !
---
Hello {{name}}{{punctuation}}
"""
    path = tmp_path / "fallback.prompt"
    path.write_text(content)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    template = load_prompt(path)

    assert template.name == "fallback_name"
    assert template.version == "v2"
    assert template.defaults == {"punctuation": "!"}
    assert template.metadata == {"audience": "analysts"}
    assert template.render(name="Ada") == "Hello Ada!"


def test_validate_invalid_variable_name():
    t = PromptTemplate(name="bad", text="Hello {{123name}}")
    issues = t.validate()
    assert any("Invalid variable name" in i for i in issues)
