import asyncio

import pytest

from sktk.agent.agent import SKTKAgent
from sktk.agent.providers import CompletionResult
from sktk.agent.router import FallbackPolicy, Router
from sktk.core.types import TokenUsage


class FlakyPrimaryProvider:
    def __init__(self, fail_every: int = 3) -> None:
        self._calls = 0
        self._fail_every = fail_every

    @property
    def name(self) -> str:
        return "primary"

    async def complete(self, messages, **kwargs):
        self._calls += 1
        if self._calls % self._fail_every == 0:
            raise RuntimeError("primary failure")
        await asyncio.sleep(0)
        return CompletionResult(
            text="primary-ok",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            metadata={"path": "primary"},
        )


class BackupProvider:
    @property
    def name(self) -> str:
        return "backup"

    async def complete(self, messages, **kwargs):
        await asyncio.sleep(0)
        return CompletionResult(
            text="backup-ok",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            metadata={"path": "backup"},
        )


class SlowProvider:
    @property
    def name(self) -> str:
        return "slow"

    async def complete(self, messages, **kwargs):
        await asyncio.sleep(0.05)
        return "slow-result"


@pytest.mark.asyncio
async def test_router_concurrency_with_flaky_primary_uses_fallback() -> None:
    router = Router(
        providers=[FlakyPrimaryProvider(fail_every=3), BackupProvider()],
        policy=FallbackPolicy(),
    )

    async def _one_call(idx: int) -> tuple[str, dict]:
        return await router.complete_with_metadata([{"role": "user", "content": f"q-{idx}"}])

    results = await asyncio.gather(*[_one_call(i) for i in range(30)])
    providers = [meta["provider"] for _, meta in results]

    assert len(results) == 30
    assert "primary" in providers
    assert "backup" in providers


@pytest.mark.asyncio
async def test_router_timeout_fallback_under_load() -> None:
    router = Router(
        providers=[SlowProvider(), BackupProvider()],
        policy=FallbackPolicy(),
    )

    async def _one_call() -> tuple[str, dict]:
        return await router.complete_with_metadata(
            [{"role": "user", "content": "timeout-test"}],
            timeout=0.005,
        )

    results = await asyncio.gather(*[_one_call() for _ in range(20)])

    assert len(results) == 20
    assert all(meta["provider"] == "backup" for _, meta in results)


@pytest.mark.asyncio
async def test_agent_cancellation_propagates() -> None:
    router = Router(
        providers=[SlowProvider(), BackupProvider()],
        policy=FallbackPolicy(),
    )
    agent = SKTKAgent(name="cancel-test", instructions="test", service=router, timeout=5.0)

    task = asyncio.create_task(agent.invoke("hello"))
    await asyncio.sleep(0.005)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
