from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from weiss_rl.runtime.components.opponents.central_opponent_groups import CentralOpponentEntry


def central_heuristic_entry(
    *,
    batch: SimpleNamespace,
    row_indices: list[int] | np.ndarray,
    obs_step: np.ndarray | None = None,
    logits_out: np.ndarray | None = None,
    values_out: np.ndarray | None = None,
) -> CentralOpponentEntry:
    rows = np.asarray(row_indices, dtype=np.int64)
    obs = obs_step if obs_step is not None else np.arange(12, dtype=np.float32).reshape(4, 3)
    return CentralOpponentEntry(
        actor=SimpleNamespace(name="actor"),
        batch=batch,
        row_indices=rows,
        obs_step=obs,
        actor_step=np.zeros((obs.shape[0],), dtype=np.int64),
        logits_out=logits_out,
        values_out=np.zeros((obs.shape[0],), dtype=np.float32) if values_out is None else values_out,
    )


class RecordingPackedHeuristicPolicy:
    def __init__(self, actions: list[int]) -> None:
        self.actions = actions
        self.calls: list[dict[str, Any]] = []

    def choose_actions_from_meta_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray:
        self.calls.append(
            {
                "obs_rows": obs_rows.copy(),
                "legal_ids": legal_ids.copy(),
                "legal_offsets": legal_offsets.copy(),
                "legal_action_meta": None if legal_action_meta is None else legal_action_meta.copy(),
            }
        )
        return np.asarray(self.actions, dtype=np.int64)
