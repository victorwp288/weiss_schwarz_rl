"""Batch field lookup for IMPALA learner internals."""

from __future__ import annotations

from typing import Any


def batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


__all__ = ["batch_value"]
