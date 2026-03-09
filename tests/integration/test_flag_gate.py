import pytest


def test_integration_flag_defaults_to_skip(monkeypatch):
    monkeypatch.delenv("SKTK_RUN_INTEGRATION", raising=False)
    from tests.integration.conftest import api_key

    with pytest.raises(pytest.skip.Exception):
        api_key()
