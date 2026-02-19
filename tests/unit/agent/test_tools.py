# tests/unit/agent/test_tools.py
import pytest

from sktk.agent.tools import Tool, _infer_parameters, tool


def test_tool_schema():
    t = Tool(
        name="search", description="Search web", fn=lambda: None, parameters={"type": "object"}
    )
    schema = t.to_schema()
    assert schema["name"] == "search"
    assert schema["description"] == "Search web"


@pytest.mark.asyncio
async def test_tool_call_sync():
    def add(a: int, b: int) -> int:
        return a + b

    t = Tool(name="add", description="Add numbers", fn=add)
    result = await t(a=2, b=3)
    assert result == 5


@pytest.mark.asyncio
async def test_tool_call_async():
    async def fetch(url: str) -> str:
        return f"fetched:{url}"

    t = Tool(name="fetch", description="Fetch URL", fn=fetch)
    result = await t(url="http://example.com")
    assert result == "fetched:http://example.com"


def test_tool_decorator():
    @tool(description="Search the web")
    def search(query: str) -> str:
        return f"results for {query}"

    assert isinstance(search, Tool)
    assert search.name == "search"
    assert search.description == "Search the web"


def test_tool_decorator_custom_name():
    @tool(name="web_search", description="Search")
    def search(query: str) -> str:
        return f"results for {query}"

    assert search.name == "web_search"


def test_infer_parameters():
    def fn(name: str, age: int, score: float = 0.0) -> None:
        pass

    params = _infer_parameters(fn)
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["age"]["type"] == "integer"
    assert params["properties"]["score"]["type"] == "number"
    assert "name" in params["required"]
    assert "age" in params["required"]
    assert "score" not in params["required"]


def test_infer_parameters_auto():
    @tool(description="Greet")
    def greet(name: str) -> str:
        return f"hello {name}"

    assert "name" in greet.parameters["properties"]
    assert greet.parameters["required"] == ["name"]


@pytest.mark.asyncio
async def test_tool_decorator_async():
    @tool(description="Async fetch")
    async def fetch(url: str) -> str:
        return f"data:{url}"

    assert isinstance(fetch, Tool)
    result = await fetch(url="test")
    assert result == "data:test"


def test_infer_parameters_skips_self_and_cls():
    def instance_fn(self, query: str) -> None:
        return None

    def class_fn(cls, limit: int) -> None:
        return None

    instance_params = _infer_parameters(instance_fn)
    class_params = _infer_parameters(class_fn)

    assert "self" not in instance_params["properties"]
    assert instance_params["required"] == ["query"]
    assert "cls" not in class_params["properties"]
    assert class_params["required"] == ["limit"]
