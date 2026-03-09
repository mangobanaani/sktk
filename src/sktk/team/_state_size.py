from __future__ import annotations

import json
from typing import Any


def _enforce_state_size(state: dict[str, Any], max_bytes: int) -> None:
    """Raise ValueError if serialized state exceeds max_bytes."""
    encoded = json.dumps(state, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"Checkpoint state exceeds limit ({len(encoded)} > {max_bytes} bytes)")
