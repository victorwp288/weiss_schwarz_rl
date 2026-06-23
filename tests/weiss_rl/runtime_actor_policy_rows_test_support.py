from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime


@dataclass
class PackedPolicyRows:
    hidden_state: torch.Tensor
    row_indices: np.ndarray
    obs_step: np.ndarray
    actor_step: np.ndarray
    legal_ids: np.ndarray
    legal_offsets: np.ndarray


class FactorizedRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.factorized_calls = 0
        self.packed_calls = 0

    def sample_factorized_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: torch.Tensor,
        pass_action_id: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat, legal_actions, sample_seeds, pass_action_id, temperature
        self.factorized_calls += 1
        batch = int(obs.shape[0])
        next_hidden = (
            torch.zeros((batch, 2, 3), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return (
            torch.full((batch,), 7, dtype=torch.long, device=obs.device),
            torch.full((batch,), -0.25, dtype=obs.dtype, device=obs.device),
            torch.full((batch,), 0.5, dtype=obs.dtype, device=obs.device),
            next_hidden,
        )

    def sample_packed_seat_aware(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del args, kwargs
        self.packed_calls += 1
        raise AssertionError("factorized runtime path should bypass packed sampler")


class ArgmaxRuntimeActorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.sample_calls = 0

    def sample_factorized_packed_seat_aware(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        del args, kwargs
        self.sample_calls += 1
        raise AssertionError("argmax opponent rows must not call the sampler")

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del obs, acting_seat, legal_actions
        logits = torch.tensor(
            [
                [0.0, 1.0, -2.0, 7.0, 3.0, -1.0, 2.0, 9.0, 5.0],
                [0.0, 8.0, -2.0, 7.0, 3.0, -1.0, 2.0, 6.0, 5.0],
            ],
            dtype=torch.float32,
            device=self.bias.device,
        )
        value = torch.full((2,), 0.5, dtype=torch.float32, device=self.bias.device)
        next_hidden = (
            torch.zeros((2, 2, 3), dtype=torch.float32, device=self.bias.device)
            if seat_hidden_state is None
            else seat_hidden_state + 1.0
        )
        return logits, value, next_hidden


def policy_rows_runtime() -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any.config = SimpleNamespace(pass_action_id=8)
    return runtime


def packed_policy_rows() -> PackedPolicyRows:
    return PackedPolicyRows(
        hidden_state=torch.zeros((2, 2, 3), dtype=torch.float32),
        row_indices=np.asarray([0, 1], dtype=np.int64),
        obs_step=np.zeros((2, 4), dtype=np.float32),
        actor_step=np.asarray([0, 1], dtype=np.int64),
        legal_ids=np.asarray([0, 7, 8, 1, 7, 8], dtype=np.uint32),
        legal_offsets=np.asarray([0, 3, 6], dtype=np.uint32),
    )


def structured_legal_meta() -> np.ndarray:
    return np.asarray(
        [
            [0, 0, 0, np.iinfo(np.uint16).max],
            [1, 0, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
            [0, 1, 0, np.iinfo(np.uint16).max],
            [1, 1, 0, np.iinfo(np.uint16).max],
            [2, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max, np.iinfo(np.uint16).max],
        ],
        dtype=np.uint16,
    )
