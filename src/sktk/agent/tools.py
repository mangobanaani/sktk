"""Tool/function registration for agents.

Agents can register callable tools that the LLM can invoke
during conversation. This is the foundation for function-calling.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Union


@dataclass(frozen=True)
class Tool:
    """A callable tool that an agent can invoke.

    Usage:
        @tool(name="search", description="Search the web")
        async def search(query: str) -> str:
            return "results..."

        agent = SKTKAgent(name="a", instructions="...", tools=[search_tool])
    """

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"

    async def __call__(self, **kwargs: Any) -> Any:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_schema(self) -> dict[str, Any]:
        """Export as a JSON-schema-like dict for LLM function-calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a Tool from a function.

    Usage:
        @tool(description="Search the web for information")
        async def search(query: str) -> str:
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        tool_params = parameters or _infer_parameters(fn)
        return Tool(
            name=tool_name,
            description=description or fn.__doc__ or "",
            fn=fn,
            parameters=tool_params,
        )

    return decorator


def _infer_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """Infer JSON-schema-style parameters from function signature."""
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            props[param_name] = {}  # no type constraint
        elif annotation in type_map:
            props[param_name] = {"type": type_map[annotation]}
        else:
            origin = typing.get_origin(annotation)
            if origin is list:
                props[param_name] = {"type": "array"}
            elif origin is dict:
                props[param_name] = {"type": "object"}
            elif origin is Union:
                args = typing.get_args(annotation)
                # Optional[X] is Union[X, None]
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and non_none[0] in type_map:
                    props[param_name] = {"type": type_map[non_none[0]]}
                else:
                    props[param_name] = {}
            else:
                props[param_name] = {"type": "string"}  # fallback for truly unknown types
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema
