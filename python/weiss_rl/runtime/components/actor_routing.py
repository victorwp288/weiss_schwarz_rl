"""Actor row-routing helpers for queue runtime collection."""

from __future__ import annotations

import numpy as np

from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


def split_focal_actor_rows(
    *,
    focal_indices: np.ndarray,
    rng: np.random.Generator,
    teacher_policy_available: bool,
    force_model_policy_lane: bool,
    heuristic_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split focal rows between model-policy and heuristic-policy actor paths."""

    focal = np.asarray(focal_indices, dtype=np.int64)
    if not bool(teacher_policy_available):
        raise RuntimeError("heuristic actor policy backend requires an initialized teacher policy")
    if bool(force_model_policy_lane):
        return focal, np.zeros((0,), dtype=np.int64)
    fraction = float(heuristic_fraction)
    if fraction >= 1.0:
        return np.zeros((0,), dtype=np.int64), focal
    if fraction <= 0.0:
        return focal, np.zeros((0,), dtype=np.int64)
    heuristic_mask = rng.random(focal.shape[0]) < fraction
    return (
        focal[~heuristic_mask].astype(np.int64, copy=False),
        focal[heuristic_mask].astype(np.int64, copy=False),
    )


def policy_train_mask_for_actor(
    *,
    focal_rows: np.ndarray,
    train_on_heuristic_actor_rows: bool,
    actor_policy_backend: str,
    force_model_policy_lane: bool,
    heuristic_fraction: float,
    opponent_policy_id_by_env: np.ndarray | None = None,
    mirror_policy_id: str = MIRROR_OPPONENT_POLICY_ID,
) -> np.ndarray:
    """Return which actor rows should contribute policy loss."""

    focal_mask = np.asarray(focal_rows, dtype=np.bool_)
    train_mask = focal_mask.copy()
    if bool(train_on_heuristic_actor_rows):
        return _include_mirror_opponent_rows(
            train_mask,
            focal_mask=focal_mask,
            opponent_policy_id_by_env=opponent_policy_id_by_env,
            mirror_policy_id=mirror_policy_id,
        )
    if str(actor_policy_backend) != "heuristic_public" or bool(force_model_policy_lane):
        return _include_mirror_opponent_rows(
            train_mask,
            focal_mask=focal_mask,
            opponent_policy_id_by_env=opponent_policy_id_by_env,
            mirror_policy_id=mirror_policy_id,
        )
    if float(heuristic_fraction) > 0.0:
        # Until row-level actor provenance is carried through the unroll, a
        # mixed heuristic/model actor schedule cannot safely train only the
        # model-sampled subset. Prefer dropping those rows over silently
        # optimizing off-policy heuristic actions as if the model sampled them.
        train_mask = np.zeros(focal_mask.shape, dtype=np.bool_)
    return _include_mirror_opponent_rows(
        train_mask,
        focal_mask=focal_mask,
        opponent_policy_id_by_env=opponent_policy_id_by_env,
        mirror_policy_id=mirror_policy_id,
    )


def _include_mirror_opponent_rows(
    train_mask: np.ndarray,
    *,
    focal_mask: np.ndarray,
    opponent_policy_id_by_env: np.ndarray | None,
    mirror_policy_id: str,
) -> np.ndarray:
    if opponent_policy_id_by_env is None:
        return train_mask
    opponent_ids = np.asarray(opponent_policy_id_by_env, dtype=object)
    if opponent_ids.shape != focal_mask.shape:
        raise ValueError(f"opponent_policy_id_by_env must have shape {focal_mask.shape}, got {opponent_ids.shape}")
    mirror_rows = (~focal_mask) & (opponent_ids == str(mirror_policy_id))
    if not bool(np.any(mirror_rows)):
        return train_mask
    output = train_mask.copy()
    output[mirror_rows] = True
    return output
