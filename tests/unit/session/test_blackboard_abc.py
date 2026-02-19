import pytest

from sktk.session.blackboard import Blackboard


def test_blackboard_is_abstract():
    with pytest.raises(TypeError):
        Blackboard()
