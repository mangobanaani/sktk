# tests/unit/session/test_redis_stub.py
import inspect

from sktk.session.backends.redis import RedisHistory
from sktk.session.history import ConversationHistory


def test_redis_history_exists():
    assert RedisHistory is not None


def test_redis_history_inherits():
    assert issubclass(RedisHistory, ConversationHistory)


def test_redis_history_has_required_methods():
    methods = ["append", "get", "clear", "fork"]
    for method in methods:
        assert hasattr(RedisHistory, method)
        assert inspect.iscoroutinefunction(getattr(RedisHistory, method))
