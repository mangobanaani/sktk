import pytest

from sktk.session.history import ConversationHistory


def test_conversation_history_is_abstract():
    with pytest.raises(TypeError):
        ConversationHistory()
