from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from weiss_rl.runtime.components.hashing import hash_state_dict, hash_unroll


def _manual_state_dict_hash(state_dict: dict[str, Any]) -> str:
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


def test_hash_unroll_hashes_actions_rewards_and_episode_seed_bytes_in_order() -> None:
    actions = np.array([[1, 2], [3, 4]], dtype=np.int32).T
    rewards = np.array([[0.5, -1.25]], dtype=np.float32)
    episode_seed = np.array([9, 10], dtype=np.uint64)
    digest = hashlib.sha256()
    for array in (actions, rewards, episode_seed):
        digest.update(np.ascontiguousarray(array).tobytes())

    assert hash_unroll(actions=actions, rewards=rewards, episode_seed=episode_seed) == digest.hexdigest()


def test_hash_state_dict_is_key_order_independent_and_matches_runtime_contract() -> None:
    first = {
        "b": torch.tensor([[1, 2]], dtype=torch.int64),
        "a": np.array([3.5], dtype=np.float32),
    }
    second = {
        "a": np.array([3.5], dtype=np.float32),
        "b": torch.tensor([[1, 2]], dtype=torch.int64),
    }

    assert hash_state_dict(first) == _manual_state_dict_hash(first)
    assert hash_state_dict(first) == hash_state_dict(second)


def test_hash_state_dict_includes_shape_even_when_raw_bytes_match() -> None:
    flat = {"weight": np.array([1, 2], dtype=np.int16)}
    matrix = {"weight": np.array([[1, 2]], dtype=np.int16)}

    assert np.ascontiguousarray(flat["weight"]).tobytes() == np.ascontiguousarray(matrix["weight"]).tobytes()
    assert hash_state_dict(flat) != hash_state_dict(matrix)
