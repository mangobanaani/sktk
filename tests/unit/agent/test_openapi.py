# tests/unit/agent/test_openapi.py
import builtins
import json
import sys
import tempfile
from types import SimpleNamespace

import pytest

from sktk.agent.openapi import tools_from_openapi, tools_from_openapi_file

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store", "version": "1.0"},
    "paths": {
        "/pets": {
            "get": {
                "operationId": "list_pets",
                "summary": "List all pets",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer"},
                    }
                ],
            },
            "post": {
                "operationId": "create_pet",
                "summary": "Create a pet",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Pet name"},
                                    "tag": {"type": "string"},
                                },
                                "required": ["name"],
                            }
                        }
                    }
                },
            },
        },
        "/pets/{petId}": {
            "get": {
                "operationId": "get_pet",
                "summary": "Get a pet by ID",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "The pet ID",
                    }
                ],
            },
        },
    },
}


def test_tools_from_openapi():
    tools = tools_from_openapi(SAMPLE_SPEC)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert "list_pets" in names
    assert "create_pet" in names
    assert "get_pet" in names


def test_tool_description():
    tools = tools_from_openapi(SAMPLE_SPEC)
    list_tool = next(t for t in tools if t.name == "list_pets")
    assert list_tool.description == "List all pets"


def test_tool_parameters():
    tools = tools_from_openapi(SAMPLE_SPEC)
    get_tool = next(t for t in tools if t.name == "get_pet")
    assert "petId" in get_tool.parameters["properties"]
    assert "petId" in get_tool.parameters["required"]


def test_tool_request_body_params():
    tools = tools_from_openapi(SAMPLE_SPEC)
    create_tool = next(t for t in tools if t.name == "create_pet")
    assert "name" in create_tool.parameters["properties"]
    assert "name" in create_tool.parameters["required"]


@pytest.mark.asyncio
async def test_tool_stub_callable():
    tools = tools_from_openapi(SAMPLE_SPEC)
    list_tool = next(t for t in tools if t.name == "list_pets")
    result = await list_tool(limit=10)
    assert result["operation"] == "list_pets"
    assert result["method"] == "get"


def test_tools_from_json_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_SPEC, f)
        f.flush()
        tools = tools_from_openapi_file(f.name)
    assert len(tools) == 3


def test_empty_spec():
    tools = tools_from_openapi({"paths": {}})
    assert tools == []


def test_schema_output():
    tools = tools_from_openapi(SAMPLE_SPEC)
    for t in tools:
        schema = t.to_schema()
        assert "name" in schema
        assert "description" in schema


def test_tools_from_yaml_file_uses_yaml_safe_load(monkeypatch):
    calls = {}

    def fake_safe_load(content):
        calls["loaded"] = content
        return SAMPLE_SPEC

    fake_yaml = SimpleNamespace(safe_load=fake_safe_load)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("openapi: 3.0.0\npaths: {}\n")
        f.flush()
        tools = tools_from_openapi_file(f.name)

    assert len(tools) == 3
    assert "openapi: 3.0.0" in calls["loaded"]


def test_tools_from_yaml_file_without_pyyaml_raises_helpful_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "yaml", raising=False)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yaml":
            raise ImportError("yaml unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("openapi: 3.0.0\npaths: {}\n")
        f.flush()
        with pytest.raises(ImportError, match="PyYAML"):
            tools_from_openapi_file(f.name)
