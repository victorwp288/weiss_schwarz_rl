from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


class AdvanceOnlyModel:
    def advance_seat_hidden(
        self,
        obs_tensor: torch.Tensor,
        actor_tensor: torch.Tensor,
        hidden_tensor: torch.Tensor,
    ) -> torch.Tensor:
        del obs_tensor, actor_tensor
        return hidden_tensor + 1.0


class FailHeuristicPolicy:
    def choose_actions_from_meta_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray:
        del obs_rows, legal_ids, legal_offsets, legal_action_meta
        raise AssertionError("python heuristic batch path should not be used")


class RecordingHeuristicPolicy:
    def __init__(self, actions: tuple[int, ...] | None = None) -> None:
        self.actions = actions
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
        if self.actions is not None:
            return np.asarray(self.actions, dtype=np.int64)
        return np.asarray(
            [int(ids_array[int(offsets_array[row_index])]) for row_index in range(offsets_array.shape[0] - 1)],
            dtype=np.int64,
        )


class NativeBasePool:
    def __init__(self, actions: tuple[int, ...]) -> None:
        self.actions = np.asarray(actions, dtype=np.uint16)
        self.calls: list[np.ndarray] = []

    def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
        indices = np.asarray(env_indices, dtype=np.uint32)
        self.calls.append(indices.copy())
        actions_out[...] = self.actions


class ProfileNativeForbiddenPool:
    profile_calls: list[tuple[np.ndarray, str]]

    def __init__(self) -> None:
        self.profile_calls = []

    def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
        del env_indices, actions_out
        raise AssertionError("base native heuristic must not be used for profiled B3/B4 opponents")

    def choose_heuristic_public_profile_actions_into(
        self,
        env_indices: np.ndarray,
        actions_out: np.ndarray,
        profile_name: str,
    ) -> None:
        del env_indices, actions_out, profile_name
        raise AssertionError("profile-native heuristic is not used until simulator/RL profile parity is proven")


class BaseNativeForbiddenPool:
    def choose_heuristic_public_actions_into(self, env_indices: np.ndarray, actions_out: np.ndarray) -> None:
        del env_indices, actions_out
        raise AssertionError("base native heuristic must not be used for profiled B3/B4 opponents")


@dataclass
class FixedOpponentRows:
    row_indices: np.ndarray
    obs_step: np.ndarray
    actor_step: np.ndarray
    legal_ids: np.ndarray
    legal_offsets: np.ndarray
    legal_action_meta: np.ndarray
    values_out: np.ndarray
    actions_out: np.ndarray
    logp_out: np.ndarray


def make_fixed_opponent_runtime_actor(
    *,
    policy_id: str,
    heuristic_policy: object,
    pool: object,
) -> tuple[QueueRuntime, Any]:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._fixed_opponent_backend = "simulator_native"
    runtime_any.action_dim = 32
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {policy_id: heuristic_policy}

    actor = SimpleNamespace(
        model=AdvanceOnlyModel(),
        compiled_model=None,
        env=SimpleNamespace(pool=pool),
        opponent_policy_id_by_env=np.asarray(
            [policy_id, policy_id, MIRROR_OPPONENT_POLICY_ID],
            dtype=object,
        ),
        seat_hidden=torch.zeros((3, 2)),
        opponent_hidden=torch.zeros((3, 2)),
    )
    return runtime, actor


def make_fixed_opponent_rows() -> FixedOpponentRows:
    return FixedOpponentRows(
        row_indices=np.asarray([0, 1], dtype=np.int64),
        obs_step=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32),
        actor_step=np.asarray([1, 1, 0], dtype=np.int64),
        legal_ids=np.asarray([10, 11, 12, 20, 21], dtype=np.uint32),
        legal_offsets=np.asarray([0, 3, 5, 5], dtype=np.uint32),
        legal_action_meta=np.asarray(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 0, 0],
                [4, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        values_out=np.ones((3,), dtype=np.float32),
        actions_out=np.full((3,), 99, dtype=np.int64),
        logp_out=np.full((3,), -1.0, dtype=np.float32),
    )


def apply_fixed_opponent_rows(runtime: QueueRuntime, *, actor: object, rows: FixedOpponentRows) -> None:
    QueueRuntime._apply_opponent_rows_ids(
        runtime,
        actor=actor,
        row_indices=rows.row_indices,
        obs_step=rows.obs_step,
        actor_step=rows.actor_step,
        legal_ids=rows.legal_ids,
        legal_offsets=rows.legal_offsets,
        legal_action_meta=rows.legal_action_meta,
        logits_out=None,
        values_out=rows.values_out,
        actions_out=rows.actions_out,
        logp_out=rows.logp_out,
        rng=np.random.default_rng(7),
        sample_actions=True,
    )


def assert_heuristic_rows_written(actor: Any, rows: FixedOpponentRows, expected_actions: list[int]) -> None:
    assert actor.seat_hidden[0].tolist() == [1.0, 1.0]
    assert actor.seat_hidden[1].tolist() == [1.0, 1.0]
    assert rows.values_out.tolist() == [0.0, 0.0, 1.0]
    assert rows.actions_out.tolist() == [*expected_actions, 99]
    assert rows.logp_out.tolist() == [0.0, 0.0, -1.0]
