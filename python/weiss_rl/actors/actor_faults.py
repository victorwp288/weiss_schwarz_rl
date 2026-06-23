"""Actor numeric-fault bundle construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.replay.bundles import write_fault_bundle


def write_actor_numeric_fault_bundle(
    *,
    fault_dir: Path,
    reason: str,
    actor_id: int,
    layout_name: str,
    update_count: int,
    observed_checkpoint_update: int,
    step: int,
    obs: np.ndarray,
    to_play: np.ndarray,
    decision_id: np.ndarray,
    episode_seed: np.ndarray,
    episode_key: np.ndarray,
    logits: np.ndarray,
    actions: np.ndarray | None = None,
    logp: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    legal_ids: np.ndarray | None = None,
    legal_offsets: np.ndarray | None = None,
    legal_mask: np.ndarray | None = None,
) -> tuple[Path, dict[str, Any]]:
    payload: dict[str, Any] = {
        "format": "numeric_fault_bundle",
        "component": "actor_worker",
        "reason": reason,
        "actor_id": int(actor_id),
        "layout_name": layout_name,
        "update_count": int(update_count),
        "observed_checkpoint_update": int(observed_checkpoint_update),
        "step": int(step),
        "obs": obs,
        "to_play": to_play,
        "decision_id": decision_id,
        "episode_seed": episode_seed,
        "episode_key": episode_key,
        "logits": logits,
        "logits_nonfinite_indices": nonfinite_indices(logits),
    }
    if actions is not None:
        payload["actions"] = actions
    if logp is not None:
        payload["logp"] = logp
        payload["logp_nonfinite_indices"] = nonfinite_indices(logp)
    if entropy is not None:
        payload["entropy"] = entropy
        payload["entropy_nonfinite_indices"] = nonfinite_indices(entropy)
    if legal_ids is not None:
        payload["legal_ids"] = legal_ids
    if legal_offsets is not None:
        payload["legal_offsets"] = legal_offsets
    if legal_mask is not None:
        payload["legal_mask"] = legal_mask

    fault_path = write_fault_bundle(
        fault_dir=fault_dir,
        prefix="actor_numeric_fault",
        payload=payload,
    )
    return fault_path, payload


def nonfinite_indices(values: np.ndarray) -> np.ndarray:
    return np.argwhere(~np.isfinite(values)).astype(np.int64, copy=False)


__all__ = [
    "nonfinite_indices",
    "write_actor_numeric_fault_bundle",
]
