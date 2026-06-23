"""Shared replay-rerun validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from weiss_rl.core.observation_layout import ObservationLayout, parse_observation_layout
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.replay.bundles import ReplayBundleMeta, ReplayStep, compute_legal_fingerprint64
from weiss_rl.replay.inspection_summaries import canonical_float


def require_single_env_batch(
    batch: DecisionBoundaryBatch,
    *,
    context: str,
    owner: str,
) -> DecisionBoundaryBatch:
    if batch.num_envs != 1:
        raise RuntimeError(f"{owner} expects a single-env batch from {context}, got {batch.num_envs}")
    return batch


def observation_dim(batch: DecisionBoundaryBatch, *, owner: str) -> int:
    obs = np.asarray(batch.obs)
    if obs.ndim != 2:
        raise RuntimeError(f"{owner} expects 2D observations, got shape {tuple(obs.shape)}")
    return int(obs.shape[1])


def require_initial_identity(*, meta: ReplayBundleMeta, batch: DecisionBoundaryBatch) -> None:
    observed_seed = int(batch.episode_seed[0])
    if observed_seed != int(meta.episode_seed64):
        raise RuntimeError(
            f"Replay reset seed mismatch: expected episode_seed64={meta.episode_seed64}, got {observed_seed}"
        )
    if meta.simulator_episode_key_u64 is None:
        return

    observed_episode_key = int(batch.episode_key[0])
    if observed_episode_key != int(meta.simulator_episode_key_u64):
        raise RuntimeError(
            "Replay reset episode_key mismatch: "
            f"expected simulator episode key {meta.simulator_episode_key_u64}, got {observed_episode_key}"
        )


def require_pre_step_match(
    *,
    step_index: int,
    expected_step: ReplayStep,
    current_batch: DecisionBoundaryBatch,
    spec_hash256: bytes,
    owner: str,
) -> None:
    observed_t = step_index
    batch_t = getattr(current_batch, "t", None)
    if batch_t is not None:
        observed_t = int(np.asarray(batch_t).reshape(-1)[0])
    if observed_t != int(expected_step.t):
        raise RuntimeError(f"Replay step index mismatch at step {step_index}")

    actual_decision_id = int(current_batch.decision_id[0])
    if actual_decision_id != int(expected_step.decision_id):
        raise RuntimeError(f"Replay decision_id mismatch at step {step_index}")

    actual_actor = int(current_batch.actor[0])
    if actual_actor != int(expected_step.actor):
        raise RuntimeError(f"Replay actor mismatch at step {step_index}")

    legal_ids = legal_ids_for_env_row(current_batch, owner=owner)
    actual_fingerprint = compute_legal_fingerprint64(
        spec_hash256=spec_hash256,
        decision_id=actual_decision_id,
        legal_ids=legal_ids,
    )
    if actual_fingerprint != int(expected_step.legal_fingerprint64):
        raise RuntimeError(f"Replay legal fingerprint mismatch at step {step_index}")


def require_post_step_match(*, step_index: int, expected_step: ReplayStep, next_batch: DecisionBoundaryBatch) -> None:
    if canonical_float(next_batch.reward[0]) != canonical_float(expected_step.reward):
        raise RuntimeError(f"Replay reward mismatch at step {step_index}")
    if bool(next_batch.terminated[0]) != bool(expected_step.terminated):
        raise RuntimeError(f"Replay terminated mismatch at step {step_index}")
    if bool(next_batch.truncated[0]) != bool(expected_step.truncated):
        raise RuntimeError(f"Replay truncated mismatch at step {step_index}")
    if int(next_batch.engine_status[0]) != int(expected_step.engine_status):
        raise RuntimeError(f"Replay engine_status mismatch at step {step_index}")


def legal_ids_for_env_row(batch: DecisionBoundaryBatch, *, owner: str) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError(f"{owner} requires ids_offsets legality in the rerun environment")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[0])
    end = int(legal_offsets[1])
    return np.asarray(legal_ids[start:end], dtype=np.uint32)


def pass_action_id_from_spec_bundle(spec_bundle: Mapping[str, Any] | None) -> int:
    if spec_bundle is None:
        return 51
    action = spec_bundle.get("action")
    if not isinstance(action, Mapping):
        return 51
    return int(action.get("pass_action_id", 51))


def load_observation_layout(spec_bundle: Mapping[str, Any] | None) -> ObservationLayout | None:
    if spec_bundle is None:
        return None
    observation = spec_bundle.get("observation")
    if not isinstance(observation, Mapping):
        return None
    try:
        return parse_observation_layout(observation)
    except (TypeError, ValueError):
        return None


__all__ = [
    "legal_ids_for_env_row",
    "load_observation_layout",
    "observation_dim",
    "pass_action_id_from_spec_bundle",
    "require_initial_identity",
    "require_post_step_match",
    "require_pre_step_match",
    "require_single_env_batch",
]
