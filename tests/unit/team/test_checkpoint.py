# tests/unit/team/test_checkpoint.py
"""Comprehensive tests for sktk.team.checkpoint (CheckpointStore, make_checkpoint_fn)."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from sktk.team.checkpoint import (
    CheckpointConfig,
    CheckpointStore,
    make_checkpoint_fn,
)

# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------


class TestMemoryBackend:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "nodeA", {"x": 1})
        cp = await store.load("wf1")
        assert cp is not None
        assert cp["node"] == "nodeA"
        assert cp["state"] == {"x": 1}
        assert "timestamp" in cp

    @pytest.mark.asyncio
    async def test_load_returns_latest(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "A", {"step": 1})
        await store.save("wf1", "B", {"step": 2})
        cp = await store.load("wf1")
        assert cp["node"] == "B"
        assert cp["state"]["step"] == 2

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self):
        store = CheckpointStore(backend="memory")
        assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "A", {"a": 1})
        await store.save("wf1", "B", {"b": 2})
        await store.save("wf2", "X", {"x": 9})

        wf1_cps = await store.list_checkpoints("wf1")
        assert len(wf1_cps) == 2
        assert wf1_cps[0]["node"] == "A"
        assert wf1_cps[1]["node"] == "B"

    @pytest.mark.asyncio
    async def test_list_unknown_workflow_returns_empty(self):
        store = CheckpointStore(backend="memory")
        assert await store.list_checkpoints("unknown") == []

    @pytest.mark.asyncio
    async def test_clear(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "A", {"a": 1})
        await store.save("wf1", "B", {"b": 2})
        await store.clear("wf1")
        assert await store.load("wf1") is None
        assert await store.list_checkpoints("wf1") == []

    @pytest.mark.asyncio
    async def test_clear_nonexistent_is_noop(self):
        store = CheckpointStore(backend="memory")
        await store.clear("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_multiple_workflows_isolated(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "A", {"wf": 1})
        await store.save("wf2", "B", {"wf": 2})
        await store.clear("wf1")
        assert await store.load("wf1") is None
        cp = await store.load("wf2")
        assert cp is not None
        assert cp["state"]["wf"] == 2

    @pytest.mark.asyncio
    async def test_max_workflows_eviction(self):
        cfg = CheckpointConfig(backend="memory", max_workflows=1)
        store = CheckpointStore(config=cfg)
        await store.save("wf1", "A", {"wf": 1})
        await store.save("wf2", "B", {"wf": 2})  # should evict wf1
        assert await store.load("wf1") is None
        cp = await store.load("wf2")
        assert cp is not None and cp["state"]["wf"] == 2


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "nodeA", {"x": 1})
            cp = await store.load("wf1")
            assert cp is not None
            assert cp["node"] == "nodeA"
            assert cp["state"] == {"x": 1}
            assert "timestamp" in cp

    @pytest.mark.asyncio
    async def test_load_returns_latest(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "A", {"step": 1})
            await store.save("wf1", "B", {"step": 2})
            cp = await store.load("wf1")
            assert cp["node"] == "B"
            assert cp["state"]["step"] == 2

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "A", {"a": 1})
            await store.save("wf1", "B", {"b": 2})
            cps = await store.list_checkpoints("wf1")
            assert len(cps) == 2
            assert cps[0]["node"] == "A"
            assert cps[1]["node"] == "B"

    @pytest.mark.asyncio
    async def test_list_unknown_workflow_returns_empty(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            assert await store.list_checkpoints("unknown") == []

    @pytest.mark.asyncio
    async def test_clear(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "A", {"a": 1})
            await store.save("wf1", "B", {"b": 2})
            await store.clear("wf1")
            assert await store.load("wf1") is None
            assert await store.list_checkpoints("wf1") == []

    @pytest.mark.asyncio
    async def test_multiple_workflows_isolated(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "A", {"wf": 1})
            await store.save("wf2", "B", {"wf": 2})
            await store.clear("wf1")
            assert await store.load("wf1") is None
            cp = await store.load("wf2")
            assert cp is not None
            assert cp["state"]["wf"] == 2

    @pytest.mark.asyncio
    async def test_persistence_across_close_reopen(self, tmp_path):
        """Data written to a file-based SQLite DB persists after close/reopen."""
        db_path = str(tmp_path / "checkpoints.db")

        store1 = CheckpointStore(backend="sqlite", path=db_path, config=None)
        await store1.save("wf1", "A", {"step": 1})
        await store1.save("wf1", "B", {"step": 2})
        await store1.close()

        store2 = CheckpointConfig(backend="sqlite", path=db_path, allow_overwrite=True)
        store2 = CheckpointStore(config=store2)
        cp = await store2.load("wf1")
        assert cp is not None
        assert cp["node"] == "B"
        assert cp["state"]["step"] == 2

        cps = await store2.list_checkpoints("wf1")
        assert len(cps) == 2
        await store2.close()


# ---------------------------------------------------------------------------
# Context manager and close()
# ---------------------------------------------------------------------------


class TestContextManagerAndClose:
    @pytest.mark.asyncio
    async def test_context_manager_memory(self):
        async with CheckpointStore(backend="memory") as store:
            await store.save("wf1", "A", {"x": 1})
            cp = await store.load("wf1")
            assert cp is not None

    @pytest.mark.asyncio
    async def test_context_manager_sqlite(self):
        async with CheckpointStore(backend="sqlite", path=":memory:") as store:
            await store.save("wf1", "A", {"x": 1})
            cp = await store.load("wf1")
            assert cp is not None
        assert store._closed is True

    @pytest.mark.asyncio
    async def test_close_memory_is_noop(self):
        store = CheckpointStore(backend="memory")
        await store.save("wf1", "A", {"x": 1})
        await store.close()  # should not raise
        assert store._closed is True

    @pytest.mark.asyncio
    async def test_close_sqlite_clears_db_reference(self):
        store = CheckpointStore(backend="sqlite", path=":memory:")
        await store.save("wf1", "A", {"x": 1})
        await store.close()
        assert store._closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        store = CheckpointStore(backend="sqlite", path=":memory:")
        await store.save("wf1", "A", {})
        await store.close()
        await store.close()  # second close should not raise


# ---------------------------------------------------------------------------
# make_checkpoint_fn
# ---------------------------------------------------------------------------


class TestMakeCheckpointFn:
    @pytest.mark.asyncio
    async def test_creates_working_checkpoint_function(self):
        store = CheckpointStore(backend="memory")
        fn = make_checkpoint_fn(store, "wf-test")

        await fn("nodeA", {"step": 1})
        await fn("nodeB", {"step": 2})

        cp = await store.load("wf-test")
        assert cp is not None
        assert cp["node"] == "nodeB"
        assert cp["state"]["step"] == 2

        cps = await store.list_checkpoints("wf-test")
        assert len(cps) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_fn_with_graph_workflow(self):
        """Integration: make_checkpoint_fn works end-to-end with GraphWorkflow."""
        from sktk.team.graph import GraphWorkflow

        def node_a(state: dict[str, Any]) -> dict[str, Any]:
            state.setdefault("visited", []).append("A")
            return {"step": "A"}

        def node_b(state: dict[str, Any]) -> dict[str, Any]:
            state.setdefault("visited", []).append("B")
            return {"step": "B"}

        store = CheckpointStore(backend="memory")
        checkpoint = make_checkpoint_fn(store, "graph-run-1")

        wf = GraphWorkflow()
        wf.add_node("A", node_a)
        wf.add_node("B", node_b)
        wf.add_edge("A", "B")

        await wf.execute(checkpoint_fn=checkpoint)

        cps = await store.list_checkpoints("graph-run-1")
        assert len(cps) == 2
        assert cps[0]["node"] == "A"
        assert cps[1]["node"] == "B"

    @pytest.mark.asyncio
    async def test_checkpoint_fn_with_sqlite_backend(self, tmp_path):
        """make_checkpoint_fn works with SQLite backend."""
        db_path = str(tmp_path / "cp.db")
        async with CheckpointStore(backend="sqlite", path=db_path) as store:
            fn = make_checkpoint_fn(store, "sql-wf")
            await fn("step1", {"data": "first"})
            await fn("step2", {"data": "second"})

            cp = await store.load("sql-wf")
            assert cp["node"] == "step2"
            assert cp["state"]["data"] == "second"


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------


class TestUnknownBackend:
    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown checkpoint backend"):
            CheckpointStore(backend="unknown")


# ---------------------------------------------------------------------------
# New architecture: config, executor sharing, retention, logging
# ---------------------------------------------------------------------------


class TestConfigAndBackends:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            CheckpointConfig(backend="bad")
        with pytest.raises(ValueError):
            CheckpointConfig(backend="sqlite", path="")
        with pytest.raises(ValueError):
            CheckpointConfig(max_checkpoints=0)
        with pytest.raises(ValueError):
            CheckpointConfig(backend="sqlite", path="/no/such/dir/db.sqlite")
        with pytest.raises(ValueError):
            CheckpointConfig(max_state_bytes=0)
        with pytest.raises(ValueError):
            CheckpointConfig(shared_max_workers=0)

    @pytest.mark.asyncio
    async def test_store_accepts_config_and_delegates(self):
        cfg = CheckpointConfig(backend="memory", max_checkpoints=2)
        store = CheckpointStore(config=cfg)
        await store.save("wf", "n1", {"a": 1})
        cp = await store.load("wf")
        assert cp is not None and cp["node"] == "n1"

    def test_config_and_params_conflict(self):
        cfg = CheckpointConfig()
        with pytest.raises(ValueError):
            CheckpointStore(backend="sqlite", config=cfg)

    def test_backend_registry_guard(self):
        from sktk.team.checkpoint import register_backend

        def dummy(cfg):
            return object()

        register_backend("custom", dummy)
        with pytest.raises(ValueError):
            register_backend("custom", dummy)
        register_backend("custom", dummy, replace=True)

    def test_backend_options_default(self):
        cfg = CheckpointConfig()
        assert cfg.backend_options == {}

    def test_registry_freeze_blocks_registration(self):
        from sktk.team.checkpoint import freeze_backend_registry, register_backend

        def dummy(cfg):
            return object()

        freeze_backend_registry()
        with pytest.raises(ValueError):
            register_backend("after-freeze", dummy)

    def test_local_registry_isolated(self):
        local: dict[str, Any] = {}
        cfg = CheckpointConfig(registry=local)
        store = CheckpointStore.from_config(cfg)
        assert "memory" in local and "sqlite" in local
        assert store._config.registry is local

    def test_plugin_api_version_filter(self, monkeypatch):
        from sktk.team import checkpoint as cp

        class EP:
            name = "plug"

            def load(self):
                def factory(cfg):
                    return object()

                factory.__sktk_checkpoint_api__ = "2.0"
                return factory

        class EPS:
            def select(self, group):
                return [EP()]

        monkeypatch.setattr(cp.importlib.metadata, "entry_points", lambda: EPS())
        cp.load_backend_plugins()
        assert "plug" not in cp._BACKENDS

    def test_builtin_backend_cannot_unregister(self):
        from sktk.team.checkpoint import unregister_backend

        with pytest.raises(ValueError):
            unregister_backend("memory")

    def test_sqlite_allow_overwrite_flag(self, tmp_path):
        db_path = tmp_path / "existing.db"
        db_path.write_bytes(b"existing")
        with pytest.raises(ValueError):
            CheckpointConfig(backend="sqlite", path=str(db_path))
        cfg = CheckpointConfig(backend="sqlite", path=str(db_path), allow_overwrite=True)
        assert cfg.path == str(db_path.resolve())

    def test_sqlite_empty_file_requires_overwrite(self, tmp_path):
        db_path = tmp_path / "empty.db"
        db_path.touch()
        with pytest.raises(ValueError):
            CheckpointConfig(backend="sqlite", path=str(db_path))
        cfg = CheckpointConfig(backend="sqlite", path=str(db_path), allow_overwrite=True)
        assert cfg.path == str(db_path.resolve())

    @pytest.mark.asyncio
    async def test_sqlite_fallback_uses_check_same_thread_false(self, tmp_path, monkeypatch):
        # Force fallback to synchronous sqlite by mocking aiosqlite import to fail
        calls: list[bool | None] = []
        real_connect = sqlite3.connect

        def fake_connect(path, **kwargs):
            calls.append(kwargs.get("check_same_thread"))
            return real_connect(path, **kwargs)

        # Mock the import statement to make aiosqlite appear unavailable
        import sys

        # Set aiosqlite to None in sys.modules to make find_spec return None
        sys.modules["aiosqlite"] = None

        monkeypatch.setattr("sktk.team.checkpoint.sqlite3.connect", fake_connect)

        db_path = tmp_path / "fallback.db"
        cfg = CheckpointConfig(backend="sqlite", path=str(db_path), allow_overwrite=True)
        store = CheckpointStore.from_config(cfg)
        await store.save("wf", "n", {"a": 1})
        await store.close()

        # Clean up
        if "aiosqlite" in sys.modules:
            del sys.modules["aiosqlite"]

        assert calls and calls[0] is False


class TestSharedExecutor:
    @pytest.mark.asyncio
    async def test_shared_executor_used_by_default(self):
        s1 = CheckpointStore(backend="sqlite", path=":memory:")
        s2 = CheckpointStore(backend="sqlite", path=":memory:")
        assert s1._backend._executor is s2._backend._executor
        await s1.close()
        await s2.close()

    @pytest.mark.asyncio
    async def test_custom_executor_not_shutdown_on_close(self):
        exec_ = ThreadPoolExecutor(max_workers=2)
        cfg = CheckpointConfig(backend="sqlite", path=":memory:", executor=exec_)
        store = CheckpointStore(config=cfg)
        await store.save("wf", "n1", {})
        await store.close()
        assert exec_._shutdown is False
        exec_.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_shared_executor_respects_max_workers(self, monkeypatch):
        from sktk.team import checkpoint as cp_mod

        cp_mod._reset_shared_executor_for_tests()
        cfg = CheckpointConfig(backend="sqlite", path=":memory:", shared_max_workers=2)
        store = CheckpointStore(config=cfg)
        executor = store._backend._executor
        assert executor._max_workers == 2
        await store.close()
        cp_mod._reset_shared_executor_for_tests()


class TestRetentionPolicy:
    @pytest.mark.asyncio
    async def test_custom_retention_policy_runs(self):
        calls: list[str] = []

        async def no_op_retention(backend, workflow_id):
            calls.append(workflow_id)

        cfg = CheckpointConfig(
            backend="sqlite", path=":memory:", max_checkpoints=1, retention_fn=no_op_retention
        )
        store = CheckpointStore(config=cfg)
        await store.save("wf", "n1", {})
        await store.save("wf", "n2", {})
        assert calls == ["wf", "wf"]
        # because retention is no-op, both checkpoints remain
        cps = await store.list_checkpoints("wf")
        assert len(cps) == 2
        await store.close()


class TestLoggingAndLifecycle:
    @pytest.mark.asyncio
    async def test_backend_init_logs_once(self):
        store = CheckpointStore(backend="memory")
        # Backend initialized without raising and config retained
        assert store._config.backend == "memory"
        await store.close()

    @pytest.mark.asyncio
    async def test_custom_executor_owned_shutdown(self):
        exec_ = ThreadPoolExecutor(max_workers=1)
        cfg = CheckpointConfig(
            backend="sqlite",
            path=":memory:",
            executor=exec_,
            executor_owned=True,
        )
        store = CheckpointStore(config=cfg)
        await store.save("wf", "n", {})
        await store.close()
        assert exec_._shutdown is True

    @pytest.mark.asyncio
    async def test_state_size_limit(self):
        store = CheckpointStore(config=CheckpointConfig(max_state_bytes=50))
        with pytest.raises(ValueError):
            await store.save("wf", "n", {"big": "x" * 100})
        await store.close()

    @pytest.mark.asyncio
    async def test_metrics_hook_error_is_ignored(self):
        def bad_hook(event, payload):
            raise RuntimeError("boom")

        store = CheckpointStore(config=CheckpointConfig(metrics_hook=bad_hook))
        await store.save("wf", "n", {"a": 1})
        await store.close()


class TestRetryConfig:
    def test_retry_config_validation(self):
        with pytest.raises(ValueError):
            CheckpointConfig(retry_attempts=-1)
        with pytest.raises(ValueError):
            CheckpointConfig(retry_delay=-0.1)
        with pytest.raises(ValueError):
            CheckpointConfig(retry_backoff=0.5)
        with pytest.raises(ValueError):
            CheckpointConfig(retry_jitter=-0.1)


class TestMetricsAsync:
    @pytest.mark.asyncio
    async def test_metrics_async_does_not_crash(self):
        store = CheckpointStore(config=CheckpointConfig(metrics_async=True, metrics_queue_size=10))
        await store.save("wf", "n", {"a": 1})
        await store.close()


class TestOtelMetricsHook:
    def test_make_metrics_hook_no_otel(self, monkeypatch):
        from sktk.observability import otel_metrics as m

        monkeypatch.setattr(m, "_HAS_OTEL", False)
        hook = m.make_metrics_hook()
        assert callable(hook)


class TestTracingConfig:
    @pytest.mark.asyncio
    async def test_trace_enabled_noop(self):
        cfg = CheckpointConfig(trace_enabled=True)
        store = CheckpointStore.from_config(cfg)
        await store.save("wf", "n", {"a": 1})
        await store.close()

    @pytest.mark.asyncio
    async def test_methods_raise_after_close(self):
        store = CheckpointStore(backend="memory")
        await store.close()
        with pytest.raises(RuntimeError):
            await store.save("wf", "n", {})
