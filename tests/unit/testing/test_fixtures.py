from sktk.testing.fixtures import (
    mock_kernel,
)
from sktk.testing.fixtures import (
    test_blackboard as make_test_blackboard,
)
from sktk.testing.fixtures import (
    test_session as make_test_session,
)


def test_mock_kernel_fixture():
    mk = mock_kernel()
    from sktk.testing.mocks import MockKernel

    assert isinstance(mk, MockKernel)


def test_test_session_fixture():
    sess = make_test_session()
    from sktk.session.session import Session

    assert isinstance(sess, Session)
    assert sess.id.startswith("test-")


def test_test_blackboard_fixture():
    bb = make_test_blackboard()
    from sktk.session.backends.memory import InMemoryBlackboard

    assert isinstance(bb, InMemoryBlackboard)
