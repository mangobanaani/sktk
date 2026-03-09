"""Typed input/output contract handling for agents."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from sktk.core.errors import ContractValidationError

T = TypeVar("T", bound=BaseModel)


def output_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a JSON schema dict for structured output."""

    schema = model.model_json_schema()
    return {
        "name": model.__name__,
        "strict": True,
        "schema": schema,
    }


def serialize_input(model: BaseModel, template: str | None = None) -> str:
    """Convert a Pydantic model to a prompt string."""
    if template is not None:
        return template.format(**model.model_dump())
    data = model.model_dump()
    lines = [f"**{k}**: {v}" for k, v in data.items()]
    return "\n".join(lines)


def parse_output(raw: str, model: type[T]) -> T:
    """Parse LLM output into a validated Pydantic model."""
    json_str = _extract_json(raw)
    if json_str is None:
        raise ContractValidationError(
            model_name=model.__name__,
            raw_output=raw,
            validation_errors=[{"loc": [], "msg": "No valid JSON found in output"}],
        )
    try:
        return model.model_validate_json(json_str)
    except ValidationError as e:
        raise ContractValidationError(
            model_name=model.__name__,
            raw_output=raw,
            validation_errors=e.errors(),  # type: ignore[arg-type]
        ) from e


def _extract_json(text: str) -> str | None:
    """Extract JSON string from raw text."""
    stripped = text.strip()

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Try extracting JSON by balanced-brace counting from each { or [
    pos = 0
    while pos < len(stripped):
        idx = _find_next_json_start(stripped, pos)
        if idx == -1:
            break
        result = _try_parse_from(stripped, idx)
        if result is not None:
            return result
        pos = idx + 1

    return None


def _find_next_json_start(text: str, start: int) -> int:
    """Find the index of the next '{' or '[' at or after *start*."""
    obj = text.find("{", start)
    arr = text.find("[", start)
    if obj == -1:
        return arr
    if arr == -1:
        return obj
    return min(obj, arr)


def _try_parse_from(text: str, start: int) -> str | None:
    """Try to parse a JSON object/array starting at the given position."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in ("{", "["):
            depth += 1
        elif c in ("}", "]"):
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    return None
    return None
