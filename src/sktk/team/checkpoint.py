"""Checkpointing support for graph workflows.

Serialize and restore workflow state to enable durable execution.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import importlib.util
import json
import logging
import os
import random
import sqlite3
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from packaging.version import InvalidVersion, Version

from sktk.observability.tracing import create_span

from ._metrics import MetricsDispatcher, MetricsHook, _emit_metrics
from ._state_size import _enforce_state_size

logger = logging.getLogger(__name__)
logger.propagate = True


_SHARED_SQLITE_EXECUTOR: ThreadPoolExecutor | None = None
_BACKENDS: dict[str, BackendFactory] = {}
_REGISTRY_FROZEN = False
_ENTRYPOINT_GROUP = "sktk.checkpoint_backends"


def _get_shared_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    global _SHARED_SQLITE_EXECUTOR
    if _SHARED_SQLITE_EXECUTOR is None:
        if max_workers is not None:
            max_workers = min(max_workers, 16)  # clamp to avoid runaway threads
            max_workers = max(max_workers, 1)
        _SHARED_SQLITE_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers or 4, thread_name_prefix="sktk-sqlite"
        )
    return _SHARED_SQLITE_EXECUTOR


def _reset_shared_executor_for_tests() -> None:
    global _SHARED_SQLITE_EXECUTOR
    if _SHARED_SQLITE_EXECUTOR is not None:
        _SHARED_SQLITE_EXECUTOR.shutdown(wait=False)
        _SHARED_SQLITE_EXECUTOR = None


_reset_shared_executor_for_tests.__doc__ = "Test-only helper; do not use in production."


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    """Register a backend factory. Intended for trusted extensions."""
    if _REGISTRY_FROZEN:
        raise ValueError("Backend registry is frozen")
    if not replace and name in _BACKENDS:
        raise ValueError(f"Backend {name!r} already registered; pass replace=True to overwrite")
    _BACKENDS[name] = factory


def unregister_backend(name: str) -> None:
    if name in ("memory", "sqlite"):
        raise ValueError("Built-in backends cannot be unregistered")
    _BACKENDS.pop(name, None)


def freeze_backend_registry() -> None:
    """Prevent further backend registrations (recommended for production)."""
    global _REGISTRY_FROZEN
    _REGISTRY_FROZEN = True


def load_backend_plugins() -> int:
    """Load backend factories from Python entry points."""
    if _REGISTRY_FROZEN:
        return 0
    count = 0
    eps = importlib.metadata.entry_points()
    for ep in eps.select(group=_ENTRYPOINT_GROUP):
        if ep.name in _BACKENDS:
            continue
        factory = ep.load()
        plugin_ver = getattr(factory, "__sktk_checkpoint_api__", None)
        if plugin_ver is not None:
            try:
                if Version(str(plugin_ver)).major != Version("1.0").major:
                    continue
            except InvalidVersion:
                continue
        register_backend(ep.name, factory)
        count += 1
    return count


class _CheckpointBackend(Protocol):
    async def save(self, workflow_id: str, node: str, state: dict[str, Any]) -> None: ...

    async def load(self, workflow_id: str) -> dict[str, Any] | None: ...

    async def list_checkpoints(self, workflow_id: str) -> list[dict[str, Any]]: ...

    async def clear(self, workflow_id: str) -> None: ...

    async def close(self) -> None: ...


# Public alias for extension authors.
CheckpointBackend = _CheckpointBackend


RetentionFn = Callable[[_CheckpointBackend, str], Awaitable[None]]


def _default_retention(max_checkpoints: int) -> RetentionFn:
    async def _enforce(backend: _CheckpointBackend, workflow_id: str) -> None:
        # No-op for memory backend (handled internally); SQLite backend trims via SQL.
        if isinstance(backend, _SQLiteBackend):
            await backend.trim(workflow_id, max_checkpoints)

    return _enforce


@dataclass
class CheckpointConfig:
    backend: str = "memory"
    path: str = ":memory:"
    max_checkpoints: int = 1000
    max_workflows: int = 1000  # memory backend total workflow cap
    max_state_bytes: int = 256_000  # per-checkpoint payload cap
    executor: ThreadPoolExecutor | None = None
    retention_fn: RetentionFn | None = None
    metrics_hook: MetricsHook | None = None
    executor_owned: bool = False
    shared_max_workers: int | None = None
    allow_overwrite: bool = False
    trace_enabled: bool = False
    trace_span_prefix: str = "sktk.checkpoint"
    backend_options: dict[str, Any] = field(default_factory=dict)
    freeze_registry: bool = False
    allow_plugin_loading: bool = True
    registry: dict[str, BackendFactory] | None = None
    plugin_api_version: str = "1.0"
    metrics_async: bool = False
    metrics_queue_size: int = 1000
    retry_attempts: int = 2
    retry_delay: float = 0.01
    retry_backoff: float = 2.0
    retry_jitter: float = 0.01

    def __post_init__(self) -> None:
        if self.backend not in ("memory", "sqlite"):
            raise ValueError(
                f"Unknown checkpoint backend {self.backend!r}, expected one of ('memory', 'sqlite')"
            )
        if self.backend == "sqlite" and not self.path:
            raise ValueError("SQLite backend requires a path")
        if self.max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be positive")
        if self.max_workflows <= 0:
            raise ValueError("max_workflows must be positive")
        if self.max_state_bytes <= 0:
            raise ValueError("max_state_bytes must be positive")
        if self.backend == "sqlite" and self.path != ":memory:":
            resolved = Path(self.path).expanduser().resolve()
            parent = resolved.parent
            if not parent.exists():
                raise ValueError(f"Directory for SQLite path does not exist: {parent}")
            if not os.access(parent, os.W_OK):
                raise ValueError(f"Directory for SQLite path is not writable: {parent}")
            if resolved.exists():
                if not os.access(resolved, os.W_OK):
                    raise ValueError(f"SQLite path is not writable: {resolved}")
                size = resolved.stat().st_size
                if not self.allow_overwrite:
                    if size == 0:
                        raise ValueError(
                            f"SQLite path exists (empty) and allow_overwrite=False: {resolved}"
                        )
                    with resolved.open("rb") as fh:
                        header = fh.read(16)
                    if header[:16] != b"SQLite format 3\x00":
                        raise ValueError(
                            f"SQLite path exists and is not a SQLite DB; set allow_overwrite=True to use: {resolved}"
                        )
            self.path = str(resolved)
        if self.shared_max_workers is not None and self.shared_max_workers < 1:
            raise ValueError("shared_max_workers must be >= 1")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        if self.retry_backoff < 1.0:
            raise ValueError("retry_backoff must be >= 1.0")
        if self.retry_jitter < 0:
            raise ValueError("retry_jitter must be >= 0")
        if self.retention_fn is None:
            self.retention_fn = _default_retention(self.max_checkpoints)
        if self.backend_options is None:
            self.backend_options = {}


# Backend factory registry interface
BackendFactory = Callable[[CheckpointConfig], "_CheckpointBackend"]


class CheckpointStore:
    """Store and retrieve workflow checkpoints.

    Supports pluggable backends for durable execution.
    """

    def __init__(
        self,
        backend: str = "memory",
        path: str = ":memory:",
        max_checkpoints: int = 1000,
        *,
        config: CheckpointConfig | None = None,
    ) -> None:
        if config and (backend != "memory" or path != ":memory:" or max_checkpoints != 1000):
            raise ValueError(
                "Provide either config or individual backend/path/max_checkpoints, not both"
            )
        cfg = config or CheckpointConfig(
            backend=backend, path=path, max_checkpoints=max_checkpoints
        )
        self._config = cfg
        registry = cfg.registry if cfg.registry is not None else _BACKENDS
        _seed_registry(registry)
        self._backend: _CheckpointBackend = _build_backend(cfg, registry)
        self._closed = False
        self._metrics_dispatcher: MetricsDispatcher | None = None
        if cfg.metrics_async:
            self._metrics_dispatcher = MetricsDispatcher(
                cfg.metrics_hook, max_queue=cfg.metrics_queue_size
            )
            # fire-and-forget; no await in __init__
            asyncio.get_event_loop().create_task(self._metrics_dispatcher.start())
        if cfg.freeze_registry:
            load_backend_plugins()
            freeze_backend_registry()

    async def save(self, workflow_id: str, node: str, state: dict[str, Any]) -> None:
        """Save a checkpoint after a node completes."""
        if self._closed:
            raise RuntimeError("CheckpointStore is closed")
        _enforce_state_size(state, self._config.max_state_bytes)
        start = time.perf_counter()
        async with _maybe_span(
            self._config,
            "save",
            {"workflow": workflow_id, "node": node},
        ):
            await self._backend.save(workflow_id, node, state)
        _emit_metrics_or_queue(
            self,
            "save",
            {
                "workflow": workflow_id,
                "node": node,
                "duration_ms": (time.perf_counter() - start) * 1000,
            },
        )

    async def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a workflow."""
        if self._closed:
            raise RuntimeError("CheckpointStore is closed")
        start = time.perf_counter()
        async with _maybe_span(self._config, "load", {"workflow": workflow_id}):
            result = await self._backend.load(workflow_id)
        _emit_metrics_or_queue(
            self,
            "load",
            {
                "workflow": workflow_id,
                "hit": result is not None,
                "duration_ms": (time.perf_counter() - start) * 1000,
            },
        )
        return result

    async def list_checkpoints(self, workflow_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a workflow."""
        if self._closed:
            raise RuntimeError("CheckpointStore is closed")
        start = time.perf_counter()
        async with _maybe_span(self._config, "list", {"workflow": workflow_id}):
            res = await self._backend.list_checkpoints(workflow_id)
        _emit_metrics_or_queue(
            self,
            "list",
            {
                "workflow": workflow_id,
                "count": len(res),
                "duration_ms": (time.perf_counter() - start) * 1000,
            },
        )
        return res

    async def clear(self, workflow_id: str) -> None:
        """Clear all checkpoints for a workflow."""
        if self._closed:
            raise RuntimeError("CheckpointStore is closed")
        start = time.perf_counter()
        async with _maybe_span(self._config, "clear", {"workflow": workflow_id}):
            await self._backend.clear(workflow_id)
        _emit_metrics_or_queue(
            self,
            "clear",
            {"workflow": workflow_id, "duration_ms": (time.perf_counter() - start) * 1000},
        )

    async def close(self) -> None:
        """Close the underlying database connection if open."""
        if not self._closed:
            start = time.perf_counter()
            async with _maybe_span(self._config, "close", {}):
                await self._backend.close()
            self._closed = True
            _emit_metrics_or_queue(
                self,
                "close",
                {"duration_ms": (time.perf_counter() - start) * 1000},
            )
            if self._metrics_dispatcher is not None:
                await self._metrics_dispatcher.stop()

    async def __aenter__(self) -> CheckpointStore:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @classmethod
    def from_config(cls, config: CheckpointConfig) -> CheckpointStore:
        """Explicit constructor for config-first usage."""
        return cls(config=config)


def make_checkpoint_fn(
    store: CheckpointStore, workflow_id: str
) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
    """Create a checkpoint callback for use with GraphWorkflow.execute()."""

    async def checkpoint(node: str, state: dict[str, Any]) -> None:
        await store.save(workflow_id, node, state)

    return checkpoint


class _MemoryBackend:
    def __init__(
        self, max_checkpoints: int, max_workflows: int, metrics_hook: MetricsHook | None = None
    ) -> None:
        self._max_checkpoints = max_checkpoints
        self._max_workflows = max_workflows
        self._memory: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._global_lock = asyncio.Lock()
        self._wf_locks: dict[str, asyncio.Lock] = {}
        self._metrics_hook = metrics_hook

    async def save(self, workflow_id: str, node: str, state: dict[str, Any]) -> None:
        checkpoint = {"node": node, "state": state, "timestamp": time.time()}
        async with self._global_lock:
            if workflow_id not in self._memory:
                self._memory[workflow_id] = []
                self._wf_locks[workflow_id] = asyncio.Lock()
                # enforce total workflows cap (drop oldest workflow entirely)
                while len(self._memory) > self._max_workflows:
                    evicted_id, _ = self._memory.popitem(last=False)
                    self._wf_locks.pop(evicted_id, None)
                    await _emit_metrics(
                        self._metrics_hook, "evict_workflow", {"workflow": evicted_id}
                    )
            wf_lock = self._wf_locks[workflow_id]
        async with wf_lock:
            checkpoints = self._memory[workflow_id]
            checkpoints.append(checkpoint)
            if len(checkpoints) > self._max_checkpoints:
                del checkpoints[: len(checkpoints) - self._max_checkpoints]

    async def load(self, workflow_id: str) -> dict[str, Any] | None:
        async with self._global_lock:
            wf_lock = self._wf_locks.get(workflow_id)
            checkpoints = self._memory.get(workflow_id, [])
        if wf_lock is None:
            return None
        async with wf_lock:
            return checkpoints[-1] if checkpoints else None

    async def list_checkpoints(self, workflow_id: str) -> list[dict[str, Any]]:
        async with self._global_lock:
            wf_lock = self._wf_locks.get(workflow_id)
            checkpoints = self._memory.get(workflow_id, [])
        if wf_lock is None:
            return []
        async with wf_lock:
            return [copy.deepcopy(c) for c in checkpoints]

    async def clear(self, workflow_id: str) -> None:
        async with self._global_lock:
            self._memory.pop(workflow_id, None)
            self._wf_locks.pop(workflow_id, None)

    async def close(self) -> None:
        # Nothing to close
        return None


class _SQLiteBackend:
    def __init__(
        self,
        path: str,
        max_checkpoints: int,
        executor: ThreadPoolExecutor | None,
        retention_fn: RetentionFn,
        executor_owned: bool,
        shared_max_workers: int | None,
        metrics_hook: MetricsHook | None,
        retry_attempts: int,
        retry_delay: float,
        retry_backoff: float,
        retry_jitter: float,
    ) -> None:
        self._path = path
        self._max_checkpoints = max_checkpoints
        self._executor = executor or _get_shared_executor(shared_max_workers)
        self._owns_executor = executor is not None and executor_owned
        self._conn: Any = None
        self._lock = asyncio.Lock()
        self._warned_fallback = False
        self._retention = retention_fn
        # retention_fn is trusted code; callers can mutate storage—documented trust boundary.
        self._metrics_hook = metrics_hook
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._retry_jitter = retry_jitter

    async def _ensure_db(self) -> Any:
        if self._conn is None:
            try:
                import aiosqlite
            except ModuleNotFoundError:  # pragma: no cover - exercised in tests
                if not self._warned_fallback:
                    logger.info("aiosqlite not installed; using synchronous sqlite fallback")
                    self._warned_fallback = True
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    self._executor,
                    lambda: sqlite3.connect(self._path, check_same_thread=False),
                )
                self._conn = _SQLiteSyncAdapter(raw, self._executor)
                await self._conn.execute("PRAGMA journal_mode=WAL")
                await self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoints "
                    "(id INTEGER PRIMARY KEY, workflow_id TEXT, data TEXT, created_at REAL)"
                )
                await self._conn.commit()
            else:
                self._conn = await aiosqlite.connect(self._path)
                await self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoints "
                    "(id INTEGER PRIMARY KEY, workflow_id TEXT, data TEXT, created_at REAL)"
                )
                await self._conn.commit()
        return self._conn

    async def save(self, workflow_id: str, node: str, state: dict[str, Any]) -> None:
        async with self._lock:
            db = await self._ensure_db()
            checkpoint = {"node": node, "state": state, "timestamp": time.time()}
            if hasattr(db, "execute"):  # both aiosqlite and sqlite3 conn
                await _execute_with_retry(
                    lambda: db.execute(
                        "INSERT INTO checkpoints (workflow_id, data, created_at) VALUES (?, ?, ?)",
                        (workflow_id, json.dumps(checkpoint), time.time()),
                    ),
                    commit=lambda: db.commit(),
                    retries=self._retry_attempts,
                    delay=self._retry_delay,
                    backoff=self._retry_backoff,
                    jitter=self._retry_jitter,
                )
            await self._retention(self, workflow_id)
            await _emit_metrics(self._metrics_hook, "save_sqlite", {"workflow": workflow_id})

    async def trim(self, workflow_id: str, max_checkpoints: int) -> None:
        db = await self._ensure_db()
        await _execute_with_retry(
            lambda: db.execute(
                "DELETE FROM checkpoints WHERE workflow_id = ? AND id NOT IN "
                "(SELECT id FROM checkpoints WHERE workflow_id = ? ORDER BY id DESC LIMIT ?)",
                (workflow_id, workflow_id, max_checkpoints),
            ),
            commit=lambda: db.commit(),
            retries=self._retry_attempts,
            delay=self._retry_delay,
            backoff=self._retry_backoff,
            jitter=self._retry_jitter,
        )
        await _emit_metrics(
            self._metrics_hook, "trim", {"workflow": workflow_id, "limit": max_checkpoints}
        )

    async def load(self, workflow_id: str) -> dict[str, Any] | None:
        async with self._lock:
            db = await self._ensure_db()
            cursor = await _execute_with_retry(
                lambda: db.execute(
                    "SELECT data FROM checkpoints WHERE workflow_id = ? ORDER BY id DESC LIMIT 1",
                    (workflow_id,),
                ),
                retries=self._retry_attempts,
                delay=self._retry_delay,
                backoff=self._retry_backoff,
                jitter=self._retry_jitter,
            )
            row = await cursor.fetchone()
            if row:
                return json.loads(row[0])  # type: ignore[no-any-return]
            return None

    async def list_checkpoints(self, workflow_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            db = await self._ensure_db()
            cursor = await _execute_with_retry(
                lambda: db.execute(
                    "SELECT data FROM checkpoints WHERE workflow_id = ? ORDER BY id ASC",
                    (workflow_id,),
                ),
                retries=self._retry_attempts,
                delay=self._retry_delay,
                backoff=self._retry_backoff,
                jitter=self._retry_jitter,
            )
            rows = await cursor.fetchall()
            return [json.loads(row[0]) for row in rows]

    async def clear(self, workflow_id: str) -> None:
        async with self._lock:
            db = await self._ensure_db()
            await _execute_with_retry(
                lambda: db.execute("DELETE FROM checkpoints WHERE workflow_id = ?", (workflow_id,)),
                commit=lambda: db.commit(),
                retries=self._retry_attempts,
                delay=self._retry_delay,
                backoff=self._retry_backoff,
                jitter=self._retry_jitter,
            )
            await _emit_metrics(self._metrics_hook, "clear_sqlite", {"workflow": workflow_id})

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
            # do not shut down shared or user-provided executors
            if self._owns_executor:
                self._executor.shutdown(wait=False)


class _SQLiteSyncAdapter:
    """Async facade over sqlite3 using a shared executor."""

    def __init__(self, conn: sqlite3.Connection, executor: ThreadPoolExecutor):
        self._conn = conn
        self._executor = executor

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> _SQLiteCursorAdapter:
        loop = asyncio.get_running_loop()
        cursor = await loop.run_in_executor(self._executor, self._conn.execute, sql, params)
        return _SQLiteCursorAdapter(cursor, self._executor)

    async def commit(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._conn.commit)

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._conn.close)


class _SQLiteCursorAdapter:
    def __init__(self, cursor: sqlite3.Cursor, executor: ThreadPoolExecutor):
        self._cursor = cursor
        self._executor = executor

    async def fetchone(self) -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._cursor.fetchone)
        finally:
            await loop.run_in_executor(self._executor, self._cursor.close)

    async def fetchall(self) -> list[Any]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._cursor.fetchall)
        finally:
            await loop.run_in_executor(self._executor, self._cursor.close)


def _build_backend(
    cfg: CheckpointConfig, registry: dict[str, BackendFactory]
) -> _CheckpointBackend:
    factory = registry.get(cfg.backend)
    if factory is None and cfg.allow_plugin_loading and registry is _BACKENDS:
        load_backend_plugins()
        factory = registry.get(cfg.backend)
    if factory is None:
        raise ValueError(f"Unknown checkpoint backend {cfg.backend!r}")
    path_kind = (
        "memory"
        if cfg.backend == "memory"
        else ("sqlite_async" if _has_aiosqlite() else "sqlite_sync")
    )
    logger.info(
        "checkpoint backend_initialized",
        extra={
            "backend": cfg.backend,
            "path": cfg.path,
            "path_kind": path_kind,
            "max_checkpoints": cfg.max_checkpoints,
            "max_workflows": cfg.max_workflows,
        },
    )
    return factory(cfg)


def _has_aiosqlite() -> bool:
    return importlib.util.find_spec("aiosqlite") is not None


class _MaybeSpan:
    def __init__(self, cfg: CheckpointConfig, name: str, attrs: dict[str, Any]) -> None:
        self._cfg = cfg
        self._name = name
        self._attrs = attrs
        self._ctx: Any = None

    async def __aenter__(self) -> Any:
        if not self._cfg.trace_enabled:
            return None
        span_name = f"{self._cfg.trace_span_prefix}.{self._name}"
        self._ctx = create_span(span_name, {k: str(v) for k, v in self._attrs.items()})
        return await self._ctx.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)


def _maybe_span(cfg: CheckpointConfig, name: str, attrs: dict[str, Any]) -> _MaybeSpan:
    return _MaybeSpan(cfg, name, attrs)


def _emit_metrics_or_queue(store: CheckpointStore, event: str, payload: dict[str, Any]) -> None:
    if store._metrics_dispatcher is not None:
        store._metrics_dispatcher.emit(event, payload)
    else:
        # fire and forget
        asyncio.get_event_loop().create_task(
            _emit_metrics(store._config.metrics_hook, event, payload)
        )


# Built-in backend registrations
def _memory_factory(cfg: CheckpointConfig) -> _CheckpointBackend:
    return _MemoryBackend(cfg.max_checkpoints, cfg.max_workflows, cfg.metrics_hook)


def _sqlite_factory(cfg: CheckpointConfig) -> _CheckpointBackend:
    return _SQLiteBackend(
        cfg.path,
        cfg.max_checkpoints,
        cfg.executor,
        cfg.retention_fn,  # type: ignore[arg-type]
        cfg.executor_owned,
        cfg.shared_max_workers,
        cfg.metrics_hook,
        cfg.retry_attempts,
        cfg.retry_delay,
        cfg.retry_backoff,
        cfg.retry_jitter,
    )


def _seed_registry(registry: dict[str, BackendFactory]) -> None:
    if "memory" not in registry:
        registry["memory"] = _memory_factory
    if "sqlite" not in registry:
        registry["sqlite"] = _sqlite_factory


_seed_registry(_BACKENDS)


async def _execute_with_retry(
    op: Callable[[], Any],
    commit: Callable[[], Any] | None = None,
    retries: int = 2,
    delay: float = 0.01,
    backoff: float = 2.0,
    jitter: float = 0.01,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = op()
            if asyncio.iscoroutine(result):
                result = await result
            if commit:
                commit_res = commit()
                if asyncio.iscoroutine(commit_res):
                    await commit_res
            return result
        except sqlite3.OperationalError as exc:  # pragma: no cover - exercised in tests
            last_exc = exc
            if attempt == retries:
                raise
            sleep_for = delay * (backoff**attempt) + random.uniform(0, jitter)  # nosec B311 - used for retry jitter, not cryptography
            await asyncio.sleep(sleep_for)
    if last_exc:
        raise last_exc
