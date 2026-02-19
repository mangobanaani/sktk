import pytest

from sktk.core.multimodal import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    wrap_input,
)
from sktk.core.types import (
    AgentName,
    Allow,
    CorrelationId,
    Deny,
    Modify,
    SessionId,
    TokenUsage,
)


def test_token_usage_defaults():
    usage = TokenUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.total_cost_usd is None


def test_token_usage_total_computed():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
    assert usage.total_tokens == 150


def test_token_usage_add():
    a = TokenUsage(prompt_tokens=100, completion_tokens=50, total_cost_usd=0.01)
    b = TokenUsage(prompt_tokens=200, completion_tokens=100, total_cost_usd=0.02)
    c = a + b
    assert c.prompt_tokens == 300
    assert c.completion_tokens == 150
    assert c.total_tokens == 450
    assert c.total_cost_usd == pytest.approx(0.03)


def test_allow_result():
    r = Allow()
    assert r.allowed is True
    assert r.reason is None


def test_deny_result():
    r = Deny(reason="blocked by policy")
    assert r.allowed is False
    assert r.reason == "blocked by policy"


def test_modify_result():
    r = Modify(content="sanitized content")
    assert r.allowed is True
    assert r.content == "sanitized content"


def test_type_aliases_are_strings():
    name: AgentName = "analyst"
    sid: SessionId = "sess-123"
    cid: CorrelationId = "corr-456"
    assert isinstance(name, str)
    assert isinstance(sid, str)
    assert isinstance(cid, str)


def test_token_usage_add_asymmetric_cost():
    """When only one side has cost, the known cost is preserved (None treated as 0.0)."""
    a = TokenUsage(prompt_tokens=100, total_cost_usd=None)
    b = TokenUsage(prompt_tokens=200, total_cost_usd=0.05)
    c = a + b
    assert c.total_cost_usd == 0.05

    d = TokenUsage(prompt_tokens=100, total_cost_usd=0.03)
    e = TokenUsage(prompt_tokens=200, total_cost_usd=None)
    f = d + e
    assert f.total_cost_usd == 0.03


# ---------------------------------------------------------------------------
# Multimodal: Message.to_dict() and wrap_input() tests
# ---------------------------------------------------------------------------


def test_message_text_only_to_dict_returns_plain_string_content():
    msg = Message(role="user", content=[TextBlock(text="hello")])
    d = msg.to_dict()
    assert d == {"role": "user", "content": "hello"}


def test_message_image_base64_to_dict():
    msg = Message(
        role="user",
        content=[
            ImageBlock(source="abc123", media_type="image/png", source_type="base64"),
        ],
    )
    d = msg.to_dict()
    assert d["role"] == "user"
    blocks = d["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[0]["source"]["data"] == "abc123"


def test_message_image_url_to_dict():
    msg = Message(
        role="user",
        content=[
            ImageBlock(
                source="https://example.com/img.png", media_type="image/png", source_type="url"
            ),
        ],
    )
    d = msg.to_dict()
    blocks = d["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "url"
    assert blocks[0]["source"]["url"] == "https://example.com/img.png"


def test_message_tool_result_to_dict_default_is_error():
    msg = Message(
        role="assistant",
        content=[
            ToolResultBlock(tool_use_id="t1", content="result text"),
        ],
    )
    d = msg.to_dict()
    blocks = d["content"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "t1"
    assert blocks[0]["content"] == "result text"
    assert blocks[0]["is_error"] is False


def test_message_tool_result_to_dict_with_is_error_true():
    msg = Message(
        role="assistant",
        content=[
            ToolResultBlock(tool_use_id="t2", content="something broke", is_error=True),
        ],
    )
    d = msg.to_dict()
    blocks = d["content"]
    assert blocks[0]["is_error"] is True


def test_wrap_input_wraps_string_as_user_message():
    msg = wrap_input("hello world")
    assert isinstance(msg, Message)
    assert msg.role == "user"
    assert msg.text() == "hello world"


def test_wrap_input_passes_through_existing_message():
    original = Message.from_text("system", "instructions")
    result = wrap_input(original)
    assert result is original
