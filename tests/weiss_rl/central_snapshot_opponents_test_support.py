from __future__ import annotations

from typing import Any

import numpy as np
from weiss_rl.runtime.components.opponents.central_opponent_groups import CentralOpponentEntry


def make_central_opponent_entry(
    *,
    actor: Any,
    batch: Any,
    row_indices: list[int],
    obs_step: np.ndarray,
    actor_step: np.ndarray,
    logits_out: np.ndarray | None,
    values_out: np.ndarray,
) -> CentralOpponentEntry:
    return CentralOpponentEntry(
        actor=actor,
        batch=batch,
        row_indices=np.asarray(row_indices, dtype=np.int64),
        obs_step=obs_step,
        actor_step=actor_step,
        logits_out=logits_out,
        values_out=values_out,
    )
