"""E2E conftest -- reuses integration fixtures."""

from __future__ import annotations

import os

import pytest

from tests.integration.conftest import *  # noqa: F401, F403

_FLAG_VALUES = {"1", "true", "yes"}


def _integration_enabled() -> bool:
    return os.environ.get("SKTK_RUN_INTEGRATION", "").lower() in _FLAG_VALUES


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _integration_enabled():
        return
    for item in items:
        if not item.nodeid.startswith("tests/e2e/"):
            continue
        item.add_marker(pytest.mark.skip(reason="SKTK_RUN_INTEGRATION not enabled"))
