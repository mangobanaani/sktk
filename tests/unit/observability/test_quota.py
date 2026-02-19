# tests/unit/observability/test_quota.py
import pytest

from sktk.agent.filters import FilterContext
from sktk.core.types import Allow, Deny
from sktk.observability.quota import TokenQuota, TokenQuotaFilter


@pytest.mark.asyncio
async def test_quota_record_and_check():
    q = TokenQuota(max_tokens=1000)
    await q.record_usage("user1", 500)
    assert await q.used("user1") == 500
    assert await q.remaining("user1") == 500
    assert not await q.is_exceeded("user1")


@pytest.mark.asyncio
async def test_quota_exceeded():
    q = TokenQuota(max_tokens=100)
    await q.record_usage("user1", 100)
    assert await q.is_exceeded("user1")
    assert await q.remaining("user1") == 0


@pytest.mark.asyncio
async def test_quota_multiple_records():
    q = TokenQuota(max_tokens=1000)
    await q.record_usage("user1", 300)
    await q.record_usage("user1", 400)
    assert await q.used("user1") == 700


@pytest.mark.asyncio
async def test_quota_separate_keys():
    q = TokenQuota(max_tokens=100)
    await q.record_usage("user1", 90)
    await q.record_usage("user2", 50)
    assert await q.used("user1") == 90
    assert await q.used("user2") == 50


@pytest.mark.asyncio
async def test_quota_filter_allows():
    q = TokenQuota(max_tokens=10000)
    f = TokenQuotaFilter(quota=q, key_field="user_id")
    ctx = FilterContext(content="hello world", stage="input", metadata={"user_id": "u1"})
    result = await f.on_input(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_quota_filter_denies():
    q = TokenQuota(max_tokens=1)
    await q.record_usage("u1", 100)
    f = TokenQuotaFilter(quota=q, key_field="user_id")
    ctx = FilterContext(content="hello", stage="input", metadata={"user_id": "u1"})
    result = await f.on_input(ctx)
    assert isinstance(result, Deny)
    assert "exceeded" in result.reason


@pytest.mark.asyncio
async def test_quota_filter_output_records():
    q = TokenQuota(max_tokens=10000)
    f = TokenQuotaFilter(quota=q, key_field="user_id")
    ctx = FilterContext(content="hello world output", stage="output", metadata={"user_id": "u1"})
    result = await f.on_output(ctx)
    assert isinstance(result, Allow)
    assert await q.used("u1") > 0


@pytest.mark.asyncio
async def test_quota_filter_function_call():
    q = TokenQuota(max_tokens=10000)
    f = TokenQuotaFilter(quota=q)
    ctx = FilterContext(content="fn", stage="function_call")
    result = await f.on_function_call(ctx)
    assert isinstance(result, Allow)
