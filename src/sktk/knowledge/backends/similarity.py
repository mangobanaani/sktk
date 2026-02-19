"""Shared vector similarity utilities for knowledge backends."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Raises ValueError when vector dimensions don't match.
    Returns 0.0 for empty or zero-norm vectors.
    """
    if len(a) != len(b):
        raise ValueError(f"cosine_similarity: dimension mismatch ({len(a)} vs {len(b)})")
    length = len(a)
    if length == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
