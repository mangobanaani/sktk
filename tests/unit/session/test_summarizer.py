# tests/unit/session/test_summarizer.py
from sktk.session.summarizer import SummaryResult, TokenBudgetSummarizer, WindowSummarizer


def _msgs(n, role="user"):
    return [{"role": role, "content": f"message {i}"} for i in range(n)]


# WindowSummarizer tests


def test_window_no_truncation():
    s = WindowSummarizer(window_size=10)
    msgs = _msgs(5)
    result = s.summarize(msgs)
    assert result.messages == msgs
    assert result.summary_text == ""
    assert result.original_count == 5
    assert result.summarized_count == 5


def test_window_truncates():
    s = WindowSummarizer(window_size=3)
    msgs = _msgs(10)
    result = s.summarize(msgs)
    # 7 dropped, kept last 3 + 1 summary message
    assert result.original_count == 10
    assert len(result.messages) == 4  # summary + 3 kept
    assert result.messages[0]["role"] == "system"
    assert "7 earlier messages" in result.messages[0]["content"]
    assert result.messages[1]["content"] == "message 7"


def test_window_preserves_system_messages():
    s = WindowSummarizer(window_size=2)
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "msg 2"},
        {"role": "user", "content": "msg 3"},
        {"role": "assistant", "content": "msg 4"},
    ]
    result = s.summarize(msgs)
    roles = [m["role"] for m in result.messages]
    # system preserved, summary added, then last 2 non-system
    assert roles[0] == "system"  # original system
    assert roles[1] == "system"  # summary
    assert len(result.messages) == 4


def test_window_no_keep_system():
    s = WindowSummarizer(window_size=2, keep_system=False)
    msgs = [
        {"role": "system", "content": "System msg"},
        {"role": "user", "content": "msg 1"},
        {"role": "user", "content": "msg 2"},
        {"role": "user", "content": "msg 3"},
    ]
    result = s.summarize(msgs)
    # System message not preserved, treated as non-system
    assert all(
        m.get("role") != "system" or "earlier" in m.get("content", "") for m in result.messages
    )


def test_window_exact_boundary():
    s = WindowSummarizer(window_size=5)
    msgs = _msgs(5)
    result = s.summarize(msgs)
    assert result.messages == msgs


# TokenBudgetSummarizer tests


def test_token_budget_no_truncation():
    s = TokenBudgetSummarizer(max_tokens=10000)
    msgs = _msgs(3)
    result = s.summarize(msgs)
    assert result.messages == msgs
    assert result.summary_text == ""


def test_token_budget_truncates():
    # Each "message X" is ~2 words * 1.3 = ~2.6 tokens
    # 10 messages = ~26 tokens, budget of 10 should truncate
    s = TokenBudgetSummarizer(max_tokens=10)
    msgs = _msgs(10)
    result = s.summarize(msgs)
    assert result.original_count == 10
    assert len(result.messages) < 10
    assert "trimmed" in result.summary_text


def test_token_budget_preserves_system():
    s = TokenBudgetSummarizer(max_tokens=20)
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "a " * 50},
        {"role": "user", "content": "b " * 50},
        {"role": "user", "content": "short"},
    ]
    result = s.summarize(msgs)
    roles = [m["role"] for m in result.messages]
    assert roles[0] == "system"  # original system preserved


def test_token_budget_keeps_recent():
    s = TokenBudgetSummarizer(max_tokens=15)
    msgs = [
        {"role": "user", "content": "old message with many words here"},
        {"role": "user", "content": "another old message"},
        {"role": "user", "content": "recent"},
    ]
    result = s.summarize(msgs)
    # Most recent messages should be kept
    kept_contents = [m["content"] for m in result.messages if m["role"] != "system"]
    assert "recent" in kept_contents


def test_token_budget_truncation_keeps_recent_messages_within_budget():
    s = TokenBudgetSummarizer(max_tokens=80, tokens_per_word=1.0)
    msgs = [{"role": "user", "content": f"msg{i} " * 20} for i in range(5)]
    result = s.summarize(msgs)
    kept = [m for m in result.messages if m["role"] != "system"]
    assert len(kept) == 1
    assert "msg4" in kept[0]["content"]
    assert "trimmed" in result.summary_text


def test_summary_result_fields():
    result = SummaryResult(
        messages=[{"role": "user", "content": "hi"}],
        original_count=5,
        summarized_count=1,
        summary_text="test",
    )
    assert result.original_count == 5
    assert result.summarized_count == 1
    assert result.summary_text == "test"
