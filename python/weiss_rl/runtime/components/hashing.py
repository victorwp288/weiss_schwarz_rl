"""Deterministic hashing helpers for runtime state fingerprints."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


def hash_unroll(*, actions: np.ndarray, rewards: np.ndarray, episode_seed: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (actions, rewards, episode_seed):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hash_state_dict(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(str(key).encode("utf-8"))
        value = state_dict[key]
        tensor = value.detach().cpu().contiguous() if torch.is_tensor(value) else torch.as_tensor(value)
        array = np.asarray(tensor.numpy())
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
