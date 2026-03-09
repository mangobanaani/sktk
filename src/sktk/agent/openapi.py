"""OpenAPI plugin generator.

Reads an OpenAPI/Swagger specification and generates Tool objects
that can be registered with an agent for automatic function calling.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from sktk.agent.tools import Tool

logger = logging.getLogger(__name__)

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_BODY_PREFIX = "body_"


def _resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a JSON ``$ref`` pointer against the root spec.

    Returns an empty dict if the reference cannot be followed.
    """
    if not ref.startswith("#/"):
        logger.warning("External $ref not supported: %s", ref)
        return {}
    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            logger.warning("Could not resolve $ref %s: missing key %r", ref, part)
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def _deref(obj: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """If *obj* contains a ``$ref`` key, resolve it; otherwise return *obj* as-is."""
    if "$ref" in obj:
        return _resolve_ref(obj["$ref"], spec)
    return obj


def _build_parameters(
    operation: dict[str, Any],
    spec: dict[str, Any],
    path_parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build JSON-schema parameters from OpenAPI operation."""
    props: dict[str, Any] = {}
    required: list[str] = []

    # Merge path-level parameters with operation-level parameters.
    # Operation-level parameters override path-level ones with the same name+in.
    merged_params: dict[tuple[str, str], dict[str, Any]] = {}
    for param in path_parameters or []:
        param = _deref(param, spec)
        key = (param.get("name", ""), param.get("in", ""))
        merged_params[key] = param
    for param in operation.get("parameters", []):
        param = _deref(param, spec)
        key = (param.get("name", ""), param.get("in", ""))
        merged_params[key] = param

    for param in merged_params.values():
        name = param.get("name", "")
        schema = _deref(param.get("schema", {}), spec)
        props[name] = {"type": schema.get("type", "string")}
        if param.get("description"):
            props[name]["description"] = param["description"]
        if param.get("required", False):
            required.append(name)

    # Handle request body
    body = _deref(operation.get("requestBody", {}), spec)
    content = body.get("content", {})
    json_body = content.get("application/json", {})
    body_schema = _deref(json_body.get("schema", {}), spec)
    if body_schema.get("properties"):
        param_names = {name for name, _in in merged_params}
        for name, prop in body_schema["properties"].items():
            prop = _deref(prop, spec)
            # Disambiguate body properties that collide with parameter names
            key = _BODY_PREFIX + name if name in param_names else name
            props[key] = {"type": prop.get("type", "string")}  # type: ignore[index]
            if prop.get("description"):
                props[key]["description"] = prop["description"]  # type: ignore[index]
        if body_schema.get("required"):
            for r in body_schema["required"]:
                required.append(_BODY_PREFIX + r if r in param_names else r)

    required = list(dict.fromkeys(required))  # deduplicate preserving order
    result: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        result["required"] = required
    return result


def _make_stub_fn(operation_id: str, method: str, path: str) -> Any:
    """Create a stub function for the tool."""

    async def stub(**kwargs: Any) -> dict[str, Any]:
        return {
            "operation": operation_id,
            "method": method,
            "path": path,
            "args": kwargs,
            "status": "stub - implement HTTP client",
        }

    stub.__name__ = operation_id
    return stub


def _make_http_fn(
    method: str,
    base_url: str,
    path: str,
    parameters: list[dict[str, Any]],
    client: httpx.AsyncClient | None = None,
) -> Any:
    """Create an async function that performs real HTTP calls via httpx.

    Parameters
    ----------
    method:
        HTTP method (``get``, ``post``, etc.).
    base_url:
        Base URL for the API, e.g. ``https://api.example.com/v1``.
    path:
        OpenAPI path template, e.g. ``/pets/{petId}``.
    parameters:
        List of resolved OpenAPI parameter objects.  Each must have at least
        ``name`` and ``in`` keys so we can classify path vs. query params.
    """

    # Pre-compute parameter classification sets for fast lookup inside the
    # hot-path of the returned closure.
    path_params: set[str] = set()
    query_params: set[str] = set()
    param_names: set[str] = set()

    for param in parameters:
        name = param.get("name", "")
        location = param.get("in", "")
        param_names.add(name)
        if location == "path":
            path_params.add(name)
        elif location == "query":
            query_params.add(name)

    async def http_call(**kwargs: Any) -> Any:  # noqa: ANN401
        kwargs = dict(kwargs)  # avoid mutating the caller's dict
        url_path = path
        for name in path_params:
            if name in kwargs:
                url_path = url_path.replace(f"{{{name}}}", str(kwargs.pop(name)))

        query: dict[str, Any] = {}
        for name in list(kwargs):
            if name in query_params:
                query[name] = kwargs.pop(name)

        # Remaining kwargs are body fields.  Strip the ``body_`` prefix that
        # ``_build_parameters`` may have added for disambiguation.
        body: dict[str, Any] = {}
        for key in list(kwargs):
            if key.startswith(_BODY_PREFIX):
                body[key[len(_BODY_PREFIX) :]] = kwargs.pop(key)
            elif key not in param_names:
                # Not a known path/query param -> body field
                body[key] = kwargs.pop(key)

        url = base_url.rstrip("/") + "/" + url_path.lstrip("/")

        async def _do_request(c: httpx.AsyncClient) -> Any:
            try:
                response = await c.request(
                    method=method.upper(),
                    url=url,
                    params=query or None,
                    json=body or None,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body_text = exc.response.text[:500]
                raise RuntimeError(
                    f"HTTP {status} from {method.upper()} {url}: {body_text}"
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"Request failed for {method.upper()} {url}: {exc}") from exc
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

        if client is not None:
            return await _do_request(client)

        async with httpx.AsyncClient() as _client:
            return await _do_request(_client)

    http_call.__name__ = re.sub(r"[{}/ ]", "_", f"{method}_{path}").strip("_")
    return http_call


def tools_from_openapi(
    spec: dict[str, Any],
    base_url: str | None = None,
    client: Any | None = None,
) -> list[Tool]:
    """Generate Tool objects from an OpenAPI specification.

    When *httpx* is installed the generated tools make real HTTP calls.
    Otherwise they return a stub dict that describes the intended request.

    Parameters
    ----------
    spec:
        Parsed OpenAPI specification (dict).
    base_url:
        Override the base URL used for HTTP requests.  When *None* the
        URL is read from ``spec["servers"][0]["url"]``.
    client:
        Optional ``httpx.AsyncClient`` instance to reuse across all
        generated tools.  When *None* (the default) each tool call
        creates a short-lived client.  Passing a shared client avoids
        the overhead of connection setup on every request.

    Usage::

        import json
        with open("openapi.json") as f:
            spec = json.load(f)
        tools = tools_from_openapi(spec, base_url="https://api.example.com")
        agent = SKTKAgent(name="api", tools=tools)
    """
    if base_url is None:
        servers = spec.get("servers", [])
        if servers:
            base_url = servers[0].get("url", "")
        else:
            base_url = ""

    use_http = _HTTPX_AVAILABLE and bool(base_url)
    if not use_http and base_url:
        logger.warning(
            "httpx is not installed; generated tools will use stubs. "
            "Install httpx for real HTTP calls: pip install httpx"
        )

    tools = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        path_parameters = path_item.get("parameters", [])
        for method, operation in path_item.items():
            if method in ("get", "post", "put", "patch", "delete"):
                op_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
                description = operation.get(
                    "summary", operation.get("description", f"{method.upper()} {path}")
                )
                parameters = _build_parameters(operation, spec, path_parameters)

                if use_http:
                    # Build the resolved parameter list so _make_http_fn
                    # can classify path vs query vs body kwargs.
                    resolved_params: list[dict[str, Any]] = []
                    merged: dict[tuple[str, str], dict[str, Any]] = {}
                    for p in path_parameters:
                        p = _deref(p, spec)
                        merged[(p.get("name", ""), p.get("in", ""))] = p
                    for p in operation.get("parameters", []):
                        p = _deref(p, spec)
                        merged[(p.get("name", ""), p.get("in", ""))] = p
                    resolved_params = list(merged.values())

                    fn = _make_http_fn(method, base_url, path, resolved_params, client=client)
                else:
                    fn = _make_stub_fn(op_id, method, path)

                tools.append(
                    Tool(
                        name=op_id,
                        description=description,
                        fn=fn,
                        parameters=parameters,
                    )
                )

    return tools


def tools_from_openapi_file(
    path: str | Path,
    base_url: str | None = None,
    client: Any | None = None,
) -> list[Tool]:
    """Load tools from an OpenAPI JSON or YAML file.

    Parameters
    ----------
    path:
        Filesystem path to an OpenAPI JSON or YAML file.
    base_url:
        Optional override for the base URL.  Forwarded to
        :func:`tools_from_openapi`.
    client:
        Optional ``httpx.AsyncClient`` instance.  Forwarded to
        :func:`tools_from_openapi`.
    """
    path = Path(path)
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("YAML support requires PyYAML: pip install pyyaml") from e
        spec = yaml.safe_load(content)
    else:
        spec = json.loads(content)

    return tools_from_openapi(spec, base_url=base_url, client=client)
