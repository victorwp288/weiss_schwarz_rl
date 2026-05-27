"""Small helper functions for actor-worker rollout collection."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from weiss_rl.core.masking import masked_logp_from_legal_ids
from weiss_rl.league.outcomes import OnlineOutcomeTracker

torch: ModuleType | None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_CHECKPOINT_METADATA_STEM = re.compile(r"(?:checkpoint_metadata|checkpoint)_(\d+)")
_OPPONENT_ID_FIELDS = (
    "opponent_policy_id_by_env",
    "opponent_policy_ids",
    "opponent_policy_id",
    "opponent_id_by_env",
    "opponent_ids",
    "opponent_id",
    "opponent_snapshot_id_by_env",
    "opponent_snapshot_ids",
    "opponent_snapshot_id",
)


def checkpoint_update_from_path(checkpoint_path: Path) -> int | None:
    match = _CHECKPOINT_METADATA_STEM.fullmatch(checkpoint_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def sampled_opponent_policy_ids(
    opponent_sampler: Any,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    sampled_policy_ids = opponent_sampler.sample(count=count, rng=rng)
    if isinstance(sampled_policy_ids, np.ndarray):
        sampled_items = sampled_policy_ids.tolist()
    else:
        sampled_items = list(sampled_policy_ids)
    if len(sampled_items) != count:
        raise ValueError(f"opponent_sampler must return {count} policy ids")
    return tuple(str(policy_id) for policy_id in sampled_items)


def actor_behavior_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def policy_logits(policy_logits_fn: Any, obs: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            out = policy_logits_fn(obs, to_play)
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        out = policy_logits_fn(obs, to_play)
    return np.asarray(out, dtype=np.float32)


def batch_to_play(batch: Any) -> np.ndarray:
    if hasattr(batch, "to_play"):
        return np.asarray(batch.to_play)
    if hasattr(batch, "to_play_seat"):
        return np.asarray(batch.to_play_seat)
    if hasattr(batch, "actor"):
        return np.asarray(batch.actor)
    raise AttributeError("batch must expose .to_play, .to_play_seat, or .actor")


def batch_reward(batch: Any) -> np.ndarray:
    if hasattr(batch, "reward"):
        return np.asarray(batch.reward)
    if hasattr(batch, "rewards"):
        return np.asarray(batch.rewards)
    raise AttributeError("batch must expose .reward or .rewards")


def batch_episode_identity(batch: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    episode_seed = getattr(batch, "episode_seed", None)
    episode_key = getattr(batch, "episode_key", None)
    seed_array = None if episode_seed is None else np.asarray(episode_seed, dtype=np.uint64)
    key_array = None if episode_key is None else np.asarray(episode_key, dtype=np.uint64)
    return seed_array, key_array


def batch_counter(batch: Any, field_name: str, *, num_envs: int) -> np.ndarray:
    values = getattr(batch, field_name, None)
    if values is None:
        return np.zeros((num_envs,), dtype=np.uint32)
    array = np.asarray(values, dtype=np.uint32)
    if array.shape != (num_envs,):
        raise ValueError(f"{field_name} must have shape ({num_envs},)")
    return array


def env_timeout_limits(env: Any) -> dict[str, int | None]:
    return {
        "max_decisions": optional_int_attr(env, "max_decisions"),
        "max_ticks": optional_int_attr(env, "max_ticks"),
        "max_no_progress_decisions": optional_int_attr(env, "max_no_progress_decisions"),
    }


def optional_int_attr(value: Any, attr_name: str) -> int | None:
    raw_value = getattr(value, attr_name, None)
    return None if raw_value is None else int(raw_value)


def episode_identity_or_zeros(identity: np.ndarray | None, *, num_envs: int) -> np.ndarray:
    if identity is None:
        return np.zeros((num_envs,), dtype=np.uint64)
    return identity


def batch_legal_mask(batch: Any) -> np.ndarray:
    if hasattr(batch, "mask"):
        return np.asarray(batch.mask)
    if hasattr(batch, "masks"):
        return np.asarray(batch.masks)
    raise AttributeError("mask layout batch must expose .mask or .masks")


def batch_legal_ids_offsets(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    ids_offsets = getattr(batch, "ids_offsets", None)
    if ids_offsets is not None:
        legal_ids, legal_offsets = ids_offsets
        return np.asarray(legal_ids), np.asarray(legal_offsets)

    if hasattr(batch, "legal_ids") and hasattr(batch, "legal_offsets"):
        return np.asarray(batch.legal_ids), np.asarray(batch.legal_offsets)

    raise AttributeError("ids_offsets layout batch must expose .ids_offsets or (.legal_ids, .legal_offsets)")


def packed_legal_ids_prefix(legal_ids: np.ndarray, legal_offsets: np.ndarray) -> np.ndarray:
    used = 0 if legal_offsets.size == 0 else int(legal_offsets[-1])
    if used < 0 or used > legal_ids.shape[0]:
        raise ValueError(f"legal_ids prefix out of bounds: used={used}, capacity={legal_ids.shape[0]}")
    return legal_ids[:used]


def refresh_opponent_ids(opponent_id_by_env: np.ndarray, *, batch: Any, env: Any, num_envs: int) -> None:
    for source in (batch, env):
        opponent_ids = extract_opponent_ids(source, num_envs=num_envs)
        if opponent_ids is None:
            continue
        known = opponent_ids != ""
        if np.any(known):
            opponent_id_by_env[known] = opponent_ids[known]


def extract_opponent_ids(source: Any, *, num_envs: int) -> np.ndarray | None:
    if source is None:
        return None
    for field_name in _OPPONENT_ID_FIELDS:
        if not hasattr(source, field_name):
            continue
        value = getattr(source, field_name)
        if value is None:
            continue
        return coerce_opponent_ids(value, num_envs=num_envs, field_name=field_name)
    return None


def coerce_opponent_ids(value: Any, *, num_envs: int, field_name: str) -> np.ndarray:
    if isinstance(value, str):
        raw_values: list[Any] = [value] * num_envs
    else:
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            raw_values = [array.item()] * num_envs
        elif array.shape == (num_envs,):
            raw_values = array.tolist()
        else:
            raise ValueError(f"{field_name} must be scalar or shape ({num_envs},)")

    opponent_ids = np.empty((num_envs,), dtype=object)
    for env_index, raw_value in enumerate(raw_values):
        opponent_ids[env_index] = "" if raw_value is None else str(raw_value).strip()
    return opponent_ids


def update_outcomes(
    tracker: OnlineOutcomeTracker,
    *,
    opponent_ids: np.ndarray,
    reward: np.ndarray,
    engine_status: np.ndarray,
    done: np.ndarray,
) -> None:
    valid_done = np.logical_and(done, np.asarray(engine_status) == 0)
    for env_index in np.flatnonzero(valid_done):
        tracker.update(
            opponent_id=str(opponent_ids[int(env_index)]),
            outcome=outcome_token_from_reward(float(reward[int(env_index)])),
        )


def outcome_token_from_reward(reward: float) -> str:
    if reward > 0.0:
        return "w"
    if reward < 0.0:
        return "l"
    return "d"
