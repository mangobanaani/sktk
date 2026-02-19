from __future__ import annotations

import json


def _enforce_state_size(state: dict, max_bytes: int) -> None:
    """Raise ValueError if serialized state exceeds max_bytes."""
    encoded = json.dumps(state, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"Checkpoint state exceeds limit ({len(encoded)} > {max_bytes} bytes)")
