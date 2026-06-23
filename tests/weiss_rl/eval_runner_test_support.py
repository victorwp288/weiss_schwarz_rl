from __future__ import annotations

from typing import Any

import numpy as np
import torch
from weiss_rl.envs.decision_env import DecisionBoundaryBatch

EPISODE_SEED = 579856027068064


def make_decision_batch(
    batch_cls: type[DecisionBoundaryBatch] = DecisionBoundaryBatch,
    *,
    terminal: bool,
    decision_id: int = 0,
) -> DecisionBoundaryBatch:
    if terminal:
        ids_offsets = (np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.int32))
        return batch_cls(
            obs=np.zeros((1, 1), dtype=np.float32),
            reward=np.zeros((1,), dtype=np.float32),
            terminated=np.array([True]),
            truncated=np.array([False]),
            to_play=np.array([-1], dtype=np.int32),
            actor=np.array([-1], dtype=np.int32),
            decision_id=np.array([decision_id], dtype=np.int64),
            engine_status=np.array([0], dtype=np.uint8),
            decision_count=np.array([decision_id], dtype=np.uint32),
            tick_count=np.array([decision_id], dtype=np.uint32),
            episode_seed=np.array([EPISODE_SEED], dtype=np.uint64),
            episode_key=np.array([1], dtype=np.uint64),
            ids_offsets=ids_offsets,
        )
    return batch_cls(
        obs=np.zeros((1, 1), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        to_play=np.array([0], dtype=np.int32),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([decision_id], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([decision_id], dtype=np.uint32),
        tick_count=np.array([decision_id], dtype=np.uint32),
        episode_seed=np.array([EPISODE_SEED], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.int32)),
    )


class FakeEvalEnv:
    def __init__(
        self,
        *,
        reset_batch: DecisionBoundaryBatch,
        step_batch: DecisionBoundaryBatch | None = None,
    ) -> None:
        self._reset_batch = reset_batch
        self._step_batch = step_batch
        self.reset_seed: int | None = None
        self.closed = False

    def reset(self, seed: int | None = None) -> DecisionBoundaryBatch:
        self.reset_seed = seed
        return self._reset_batch

    def step(self, actions: np.ndarray) -> DecisionBoundaryBatch:
        if self._step_batch is None:
            raise AssertionError("FakeEvalEnv.step called without a step batch")
        return self._step_batch

    def close(self) -> None:
        self.closed = True


class RecordingEvalModel:
    def __init__(self) -> None:
        self.scoring_modes: list[str] = []

    def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, 1), device=device)

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.scoring_modes.append(str(scoring_mode))
        logits = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
        values = torch.zeros((1,), dtype=torch.float32, device=obs.device)
        next_hidden = torch.zeros((1, 1), dtype=torch.float32, device=obs.device)
        return logits, values, next_hidden


def make_scheduled_game(
    game_cls: type[Any],
    *,
    focal_policy_id: str,
    opponent_policy_id: str = "baseline",
    seat0_policy_id: str | None = None,
    seat1_policy_id: str | None = None,
) -> Any:
    return game_cls(
        pair_index=0,
        swap_index=0,
        episode_index=0,
        episode_seed=EPISODE_SEED,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id or focal_policy_id,
        seat1_policy_id=seat1_policy_id or opponent_policy_id,
        focal_seat=0,
    )
