from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.runtime import QueueRuntime

SNAPSHOT_POLICY_ID = "policy_000007"


class ConstantSeatAwareOpponentModel:
    def __init__(self, *, logit_value: float, value: float, hidden_increment: float) -> None:
        self.logit_value = float(logit_value)
        self.value = float(value)
        self.hidden_increment = float(hidden_increment)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def forward_seat_aware(
        self,
        obs_tensor: torch.Tensor,
        actor_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(
            (
                obs_tensor.detach().cpu().numpy().copy(),
                actor_tensor.detach().cpu().numpy().copy(),
            )
        )
        logits = torch.full((obs_tensor.shape[0], 5), self.logit_value, dtype=torch.float32)
        values = torch.full((obs_tensor.shape[0],), self.value, dtype=torch.float32)
        return logits, values, hidden_tensor + self.hidden_increment


class SequentialSeatAwareOpponentModel:
    def __init__(self, *, hidden_increment: float) -> None:
        self.hidden_increment = float(hidden_increment)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def forward_seat_aware(
        self,
        obs_tensor: torch.Tensor,
        actor_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(
            (
                obs_tensor.detach().cpu().numpy().copy(),
                actor_tensor.detach().cpu().numpy().copy(),
            )
        )
        logits = torch.arange(obs_tensor.shape[0] * 5, dtype=torch.float32).reshape(obs_tensor.shape[0], 5)
        values = torch.arange(obs_tensor.shape[0], dtype=torch.float32) + 10.0
        return logits, values, hidden_tensor + self.hidden_increment


class AdvanceOnlyModel:
    def advance_seat_hidden(
        self,
        obs_tensor: torch.Tensor,
        actor_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
    ) -> torch.Tensor:
        del obs_tensor, actor_tensor
        return hidden_tensor + 1.0


class RecordingHeuristicPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []

    def choose_actions_from_meta_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray:
        obs_array = np.asarray(obs_rows, dtype=np.int32)
        ids_array = np.asarray(legal_ids, dtype=np.uint32)
        offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
        meta_array = None if legal_action_meta is None else np.asarray(legal_action_meta, dtype=np.uint16)
        self.calls.append(
            (
                obs_array.copy(),
                ids_array.copy(),
                offsets_array.copy(),
                None if meta_array is None else meta_array.copy(),
            )
        )
        return np.asarray(
            [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
            dtype=np.int64,
        )


def central_opponent_runtime(
    *,
    snapshot_model: object | None = None,
    heuristic_policies: dict[str, object] | None = None,
    action_dim: int = 0,
) -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._profile_timers = False
    runtime_any._fixed_opponent_backend = "python_batched"
    runtime_any.action_dim = int(action_dim)
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    if snapshot_model is not None:
        runtime_any._opponent_models = {SNAPSHOT_POLICY_ID: snapshot_model}
        runtime_any._opponent_model_locks = {SNAPSHOT_POLICY_ID: threading.Lock()}
    runtime_any._opponent_heuristic_policies = dict(heuristic_policies or {})
    return runtime


def central_actor(
    *,
    focal_seats: list[int],
    opponent_policy_ids: list[str],
    hidden_width: int,
    **extra_fields: object,
) -> Any:
    return SimpleNamespace(
        focal_seat_by_env=np.asarray(focal_seats, dtype=np.int64),
        opponent_policy_id_by_env=np.asarray(opponent_policy_ids, dtype=object),
        opponent_hidden=torch.zeros((len(opponent_policy_ids), hidden_width)),
        **extra_fields,
    )
