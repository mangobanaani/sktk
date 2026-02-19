# tests/unit/agent/test_permissions.py
import pytest

from sktk.agent.filters import FilterContext
from sktk.agent.permissions import PermissionPolicy, RateLimitPolicy
from sktk.core.types import Allow, Deny
from sktk.observability.audit import AuditTrail


@pytest.fixture
def allow_policy():
    return PermissionPolicy(allow=["search", "calculate"])


@pytest.fixture
def deny_policy():
    return PermissionPolicy(deny=["delete_db", "send_email"])


@pytest.fixture
def audit_trail():
    return AuditTrail()


@pytest.mark.asyncio
async def test_allowlist_permits(allow_policy):
    ctx = FilterContext(
        content="search", stage="function_call", metadata={"function_name": "search"}
    )
    result = await allow_policy.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_allowlist_denies(allow_policy):
    ctx = FilterContext(
        content="delete", stage="function_call", metadata={"function_name": "delete"}
    )
    result = await allow_policy.on_function_call(ctx)
    assert isinstance(result, Deny)
    assert "not in allowed list" in result.reason


@pytest.mark.asyncio
async def test_denylist_permits(deny_policy):
    ctx = FilterContext(
        content="search", stage="function_call", metadata={"function_name": "search"}
    )
    result = await deny_policy.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_denylist_denies(deny_policy):
    ctx = FilterContext(
        content="delete_db", stage="function_call", metadata={"function_name": "delete_db"}
    )
    result = await deny_policy.on_function_call(ctx)
    assert isinstance(result, Deny)
    assert "explicitly denied" in result.reason


@pytest.mark.asyncio
async def test_no_lists_permits_everything():
    policy = PermissionPolicy()
    ctx = FilterContext(
        content="anything", stage="function_call", metadata={"function_name": "anything"}
    )
    result = await policy.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_on_input_always_allows():
    policy = PermissionPolicy(allow=["search"])
    ctx = FilterContext(content="hello", stage="input")
    result = await policy.on_input(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_on_output_always_allows():
    policy = PermissionPolicy(deny=["bad"])
    ctx = FilterContext(content="result", stage="output")
    result = await policy.on_output(ctx)
    assert isinstance(result, Allow)


def test_is_allowed_with_allowlist(allow_policy):
    assert allow_policy.is_allowed("search") is True
    assert allow_policy.is_allowed("delete") is False


def test_is_allowed_with_denylist(deny_policy):
    assert deny_policy.is_allowed("search") is True
    assert deny_policy.is_allowed("delete_db") is False


@pytest.mark.asyncio
async def test_function_name_falls_back_to_content():
    policy = PermissionPolicy(allow=["search"])
    ctx = FilterContext(content="search", stage="function_call")
    result = await policy.on_function_call(ctx)
    assert isinstance(result, Allow)


# RateLimitPolicy tests


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    policy = RateLimitPolicy(max_calls=5, window_seconds=60.0)
    ctx = FilterContext(content="test", stage="input")
    for _ in range(5):
        result = await policy.on_input(ctx)
        assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_rate_limit_denies_over_limit():
    policy = RateLimitPolicy(max_calls=2, window_seconds=60.0)
    ctx = FilterContext(content="test", stage="input")
    await policy.on_input(ctx)
    await policy.on_input(ctx)
    result = await policy.on_input(ctx)
    assert isinstance(result, Deny)
    assert "Rate limit exceeded" in result.reason


@pytest.mark.asyncio
async def test_rate_limit_on_function_call_always_allows():
    policy = RateLimitPolicy(max_calls=1, window_seconds=60.0)
    ctx = FilterContext(content="fn", stage="function_call")
    result = await policy.on_function_call(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_rate_limit_on_output_always_allows():
    policy = RateLimitPolicy(max_calls=1, window_seconds=60.0)
    ctx = FilterContext(content="out", stage="output")
    result = await policy.on_output(ctx)
    assert isinstance(result, Allow)


@pytest.mark.asyncio
async def test_permission_policy_logs_denial(audit_trail):
    policy = PermissionPolicy(deny=["delete_db"], audit_trail=audit_trail)
    ctx = FilterContext(
        content="delete_db",
        stage="function_call",
        metadata={"function_name": "delete_db", "session_id": "sess-1"},
    )
    result = await policy.on_function_call(ctx)
    assert isinstance(result, Deny)
    entries = await audit_trail.query(action="permission_denied")
    assert entries
    assert entries[-1].details["function_name"] == "delete_db"


@pytest.mark.asyncio
async def test_rate_limit_policy_logs_denial(audit_trail):
    policy = RateLimitPolicy(max_calls=1, window_seconds=60.0, audit_trail=audit_trail)
    ctx = FilterContext(content="test", stage="input", metadata={"session_id": "sess-2"})
    await policy.on_input(ctx)
    await policy.on_input(ctx)
    entries = await audit_trail.query(action="rate_limit_exceeded")
    assert len(entries) == 1
    assert "reason" in entries[0].details
