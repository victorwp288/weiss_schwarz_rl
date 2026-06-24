"""Small batch access helpers for learner-side factorized policy paths."""

from __future__ import annotations

from typing import Any


def factorized_batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


__all__ = ["factorized_batch_value"]
