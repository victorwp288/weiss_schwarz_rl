from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.action_logp import packed_scores_action_logp_and_entropy
from weiss_rl.learners.impala_learner import (
    ImpalaLearner,
    _chosen_action_outcome_metrics,
    _masked_action_logp_and_entropy,
    _packed_structured_legal_view,
    compute_structured_teacher_auxiliary_metrics,
    summarize_structured_policy_metrics,
)
from weiss_rl.learners.structured_teacher_auxiliary import (
    compute_structured_teacher_auxiliary_metrics as compute_structured_teacher_auxiliary_metrics_impl,
)
from weiss_rl.learners.vtrace import VTraceTargets


class NaNLogitModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.bias.unsqueeze(0).expand(batch, -1).clone()
        logits[0, 0] = torch.nan
        values = torch.zeros(batch, dtype=obs.dtype, device=obs.device)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


def test_impala_learner_reexports_structured_teacher_auxiliary_metrics() -> None:
    assert compute_structured_teacher_auxiliary_metrics is compute_structured_teacher_auxiliary_metrics_impl


class NaNGradientModel(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(action_dim))
        self.value_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(obs.shape[0])
        logits = self.logit_bias.unsqueeze(0).expand(batch, -1).clone()
        logits.register_hook(lambda grad: torch.full_like(grad, torch.nan))
        values = self.value_bias.expand(batch)
        next_hidden = torch.zeros((batch, 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class TinyPolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int = 2, action_dim: int = 3) -> None:
        super().__init__()
        self.policy = nn.Linear(observation_dim, action_dim)
        self.value = nn.Linear(observation_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        values = self.value(obs).squeeze(-1)
        next_hidden = torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class SeatAwareTinyPolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int = 2, action_dim: int = 2) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(observation_dim, 1)

    def _next_hidden(
        self,
        obs: torch.Tensor,
        seat_hidden_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if seat_hidden_state is None:
            return torch.zeros((int(obs.shape[0]), 2, 1), dtype=obs.dtype, device=obs.device)
        return seat_hidden_state + 1.0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del acting_seat, legal_actions
        batch = int(obs.shape[0])
        logits = self.logit_bias.unsqueeze(0).expand(batch, -1)
        values = self.value(obs).squeeze(-1)
        return logits, values, self._next_hidden(obs, seat_hidden_state)

    def forward_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del legal_actions
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, next_hidden = self.forward_seat_aware(
            flat_obs,
            acting_seat.reshape(time_steps * batch_size),
            seat_hidden_state,
        )
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )

    def value_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del acting_seat, seat_hidden_state
        return self.value(obs).squeeze(-1)


class TinyStructuredTeacherModel(nn.Module):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__()
        self.action_catalog = action_catalog
        self.supports_legal_candidate_scoring = True
        self.policy = nn.Linear(observation_dim, action_catalog.action_space_size)
        self.value = nn.Linear(observation_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        if legal_actions is not None:
            legal_mask = torch.as_tensor(
                legal_actions.to_mask(expected_shape=(1, int(obs.shape[0])), action_space=logits.shape[-1])[0]
            )
            logits = torch.where(legal_mask.to(device=logits.device), logits, torch.full_like(logits, -1.0e9))
        values = self.value(obs).squeeze(-1)
        next_hidden = torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


class ForwardProxyModel(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.forward_calls = 0

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        return self.base(obs, hidden_state)


class SequenceStructuredTeacherModel(TinyStructuredTeacherModel):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__(action_catalog, observation_dim=observation_dim)
        self.sequence_calls = 0
        self.step_calls = 0

    def forward_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.step_calls += 1
        hidden_state = (
            torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
            if seat_hidden_state is None
            else seat_hidden_state
        )
        return self.forward(obs, hidden_state, legal_actions=legal_actions)

    def forward_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sequence_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, _next_hidden = self.forward(flat_obs, None, legal_actions=legal_actions)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )


class TrunkStructuredTeacherModel(TinyStructuredTeacherModel):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__(action_catalog, observation_dim=observation_dim)
        self.trunk_calls = 0
        self.scorer_calls = 0
        self.sequence_calls = 0
        self.scorer_row_count = 0
        self.scorer_candidate_count = 0

    def forward_trunk_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        self.trunk_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        values = self.value(flat_obs).squeeze(-1).reshape(time_steps, batch_size)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return flat_obs, flat_obs, {"flat_obs": flat_obs}, values, next_hidden

    def score_packed_legal_candidates(
        self,
        recurrent_outputs: torch.Tensor,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch,
        *,
        state_repr: torch.Tensor | None = None,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        self.scorer_calls += 1
        assert state_repr is not None
        assert observation_context is not None
        self.scorer_row_count = int(obs.shape[0])
        ids = torch.as_tensor(legal_actions.ids, device=obs.device, dtype=torch.long)
        self.scorer_candidate_count = int(ids.shape[0])
        offsets = torch.as_tensor(legal_actions.offsets, device=obs.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(int(lengths.shape[0]), device=obs.device, dtype=torch.long),
            lengths,
        )
        flat_logits = self.policy(obs)
        return flat_logits[row_indices, ids]

    def forward_sequence_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sequence_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        flat_logits, flat_values, _next_hidden = self.forward(flat_obs, None, legal_actions=legal_actions)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return (
            flat_logits.reshape(time_steps, batch_size, -1),
            flat_values.reshape(time_steps, batch_size),
            next_hidden,
        )


class FactorizedStructuredTeacherModel(nn.Module):
    def __init__(self, action_catalog: ActionCatalog, observation_dim: int = 2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.action_catalog = action_catalog
        self.supports_legal_candidate_scoring = True
        self.supports_factorized_legal_policy = True
        self.factorized_calls = 0
        self.trunk_calls = 0
        self.public_student_calls = 0
        self.public_target_calls = 0
        self.public_target_profiles: list[str] = []
        self.factorized_candidate_logp_calls = 0

    @property
    def policy_head(self) -> Any:
        return self

    def forward_trunk_sequence_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        del acting_seat, seat_hidden_state
        self.trunk_calls += 1
        time_steps, batch_size, obs_dim = obs.shape
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        values = self.bias.expand(time_steps, batch_size)
        next_hidden = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        return flat_obs, flat_obs, {"flat_obs": flat_obs}, values, next_hidden

    def score_packed_legal_candidates(
        self,
        recurrent_outputs: torch.Tensor,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        *,
        state_repr: torch.Tensor | None = None,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> torch.Tensor:
        del recurrent_outputs, state_repr, observation_context, scoring_mode
        self.public_student_calls += 1
        action_dim = int(self.action_catalog.action_space_size)
        flat_logits = torch.full((obs.shape[0], action_dim), -20.0, dtype=obs.dtype, device=obs.device)
        flat_logits[:, 0] = self.bias + 0.5
        flat_logits[:, 5] = self.bias + 4.0
        flat_logits[:, self.action_catalog.pass_action_id] = self.bias - 5.0
        ids = torch.as_tensor(legal_actions.ids, device=obs.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=obs.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(int(lengths.shape[0]), device=obs.device, dtype=torch.long),
            lengths,
        )
        return flat_logits[row_indices, ids]

    def score_packed_public_heuristic_candidates(
        self,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        *,
        observation_context: dict[str, torch.Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> torch.Tensor:
        del obs, observation_context
        self.public_target_calls += 1
        self.public_target_profiles.append(str(scoring_profile))
        ids = torch.as_tensor(legal_actions.ids, dtype=torch.long)
        logits = torch.full((int(ids.shape[0]),), -6.0, dtype=self.bias.dtype, device=self.bias.device)
        if str(scoring_profile) == "aggressive":
            logits = torch.where(ids == 5, self.bias + 4.5, logits)
            logits = torch.where(ids == 0, self.bias - 0.5, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 7.0, logits)
        elif str(scoring_profile) == "control":
            logits = torch.where(ids == 5, self.bias + 1.5, logits)
            logits = torch.where(ids == 0, self.bias + 2.0, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 5.0, logits)
        else:
            logits = torch.where(ids == 5, self.bias + 3.0, logits)
            logits = torch.where(ids == 0, self.bias + 0.0, logits)
            logits = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 6.0, logits)
        return logits

    def factorized_packed_action_log_probs(
        self,
        latent: torch.Tensor,
        *,
        obs: torch.Tensor,
        legal_actions: LegalActionBatch | Any,
        observation_context: dict[str, torch.Tensor] | None = None,
        state_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del latent, obs, observation_context, state_repr
        self.factorized_candidate_logp_calls += 1
        ids = torch.as_tensor(legal_actions.ids, dtype=torch.long, device=self.bias.device)
        offsets = torch.as_tensor(legal_actions.offsets, dtype=torch.long, device=self.bias.device)
        raw_scores = torch.full((int(ids.shape[0]),), -6.0, dtype=self.bias.dtype, device=self.bias.device)
        raw_scores = torch.where(ids == 0, self.bias + 4.0, raw_scores)
        raw_scores = torch.where(ids == 11, self.bias + 4.0, raw_scores)
        raw_scores = torch.where(ids == int(self.action_catalog.pass_action_id), self.bias - 7.0, raw_scores)
        row_count = int(offsets.shape[0] - 1)
        log_probs = torch.empty_like(raw_scores)
        for row_index in range(row_count):
            start = int(offsets[row_index].item())
            end = int(offsets[row_index + 1].item())
            log_probs[start:end] = torch.log_softmax(raw_scores[start:end], dim=0)
        return log_probs

    def evaluate_factorized_sequence_packed_seat_aware(
        self,
        obs: torch.Tensor,
        acting_seat: torch.Tensor,
        seat_hidden_state: torch.Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | Any,
        actions: torch.Tensor | None = None,
        same_family_reference_actions: torch.Tensor | None = None,
        same_family_reference_families: torch.Tensor | None = None,
    ) -> Any:
        del legal_actions, acting_seat, seat_hidden_state, same_family_reference_families
        self.factorized_calls += 1
        time_steps, batch_size, _obs_dim = obs.shape
        family_index = {family.name: index for index, family in enumerate(self.action_catalog.families)}
        attack_type_index = {name: index for index, name in enumerate(self.action_catalog.attack_type_names)}
        family_count = len(self.action_catalog.families)
        max_stage = max(int(self.action_catalog.max_stage), 1)
        attack_slot_count = max(int(self.action_catalog.attack_slot_count), 1)
        attack_type_count = max(len(self.action_catalog.attack_type_names), 1)

        family_logits = torch.full(
            (time_steps, batch_size, family_count),
            -2.0,
            dtype=obs.dtype,
            device=obs.device,
        )
        family_logits[0, 0, family_index["main_play_character"]] = self.bias + 4.0
        if time_steps * batch_size > 1 and "attack" in family_index:
            family_logits.reshape(-1, family_count)[1, family_index["attack"]] = self.bias + 4.0
        family_log_probs = torch.log_softmax(family_logits, dim=-1)

        play_slot_logits = torch.zeros((time_steps, batch_size, max_stage), dtype=obs.dtype, device=obs.device)
        play_slot_logits[..., 0] = self.bias + 3.0
        play_slot_log_probs = torch.log_softmax(play_slot_logits, dim=-1)

        attack_slot_logits = torch.zeros(
            (time_steps, batch_size, attack_slot_count), dtype=obs.dtype, device=obs.device
        )
        attack_slot_logits[..., 0] = self.bias + 3.0
        attack_slot_log_probs = torch.log_softmax(attack_slot_logits, dim=-1)

        attack_type_logits = torch.zeros(
            (time_steps, batch_size, attack_type_count), dtype=obs.dtype, device=obs.device
        )
        attack_type_logits[..., attack_type_index.get("direct", 0)] = self.bias + 3.0
        attack_type_log_probs = torch.log_softmax(attack_type_logits, dim=-1)

        values = self.bias.expand(time_steps, batch_size)
        entropy = torch.ones((time_steps, batch_size), dtype=obs.dtype, device=obs.device) * (self.bias + 1.0)
        action_logp = None if actions is None else self.bias.expand(time_steps, batch_size) - 0.25
        top_action_ids = None
        same_family_action_logp = None
        same_family_top_action_ids = None
        same_family_arg0_logp = None
        same_family_top_arg0 = None
        if same_family_reference_actions is not None:
            same_family_action_logp = self.bias.expand(time_steps, batch_size) - 0.1
            same_family_top_action_ids = same_family_reference_actions.to(device=obs.device, dtype=torch.long)
            top_action_ids = same_family_reference_actions.to(device=obs.device, dtype=torch.long)
            flat_top_arg0 = []
            for action_id in same_family_reference_actions.reshape(-1).tolist():
                decoded = self.action_catalog.decode(int(action_id))
                flat_top_arg0.append(-1 if decoded.hand_index is None else int(decoded.hand_index))
            same_family_top_arg0 = torch.as_tensor(flat_top_arg0, dtype=torch.long, device=obs.device).reshape(
                time_steps,
                batch_size,
            )
            same_family_arg0_logp = self.bias.expand(time_steps, batch_size) - 0.05
        return SimpleNamespace(
            values=values,
            action_logp=action_logp,
            entropy=entropy,
            family_log_probs=family_log_probs,
            play_slot_log_probs=play_slot_log_probs,
            move_source_log_probs=play_slot_log_probs,
            move_slot_log_probs=play_slot_log_probs,
            attack_slot_log_probs=attack_slot_log_probs,
            attack_type_log_probs=attack_type_log_probs,
            top_action_ids=top_action_ids,
            same_family_action_logp=same_family_action_logp,
            same_family_top_action_ids=same_family_top_action_ids,
            same_family_arg0_logp=same_family_arg0_logp,
            same_family_top_arg0=same_family_top_arg0,
        )


class _ScaledLoss:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss

    def backward(self) -> None:
        self.loss.backward()


class FakeGradScaler:
    def __init__(self, *, scale: float = 8.0, overflow: bool = False) -> None:
        self.scale_value = float(scale)
        self.overflow = overflow

    def get_scale(self) -> float:
        return self.scale_value

    def scale(self, loss: torch.Tensor) -> _ScaledLoss:
        return _ScaledLoss(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.overflow:
            optimizer.step()

    def update(self, new_scale: float | None = None) -> None:
        if new_scale is not None:
            self.scale_value = float(new_scale)
        elif self.overflow:
            self.scale_value *= 0.5


def _simple_training_batch() -> dict[str, object]:
    return {
        "obs": np.asarray(
            [
                [[1.0, 0.0]],
                [[0.5, -0.5]],
            ],
            dtype=np.float32,
        ),
        "actions": np.asarray(
            [
                [0],
                [1],
            ],
            dtype=np.int64,
        ),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }


def _packed_ids_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    offsets = [0]
    for row in np.asarray(mask, dtype=bool).reshape(-1, mask.shape[-1]):
        row_ids = np.flatnonzero(row).astype(np.uint32)
        ids.extend(int(value) for value in row_ids.tolist())
        offsets.append(len(ids))
    return np.asarray(ids, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32)


def _packed_meta_from_ids(action_catalog: ActionCatalog, packed_ids: np.ndarray) -> np.ndarray:
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    unused = np.iinfo(np.uint16).max
    rows = np.full((int(packed_ids.shape[0]), 4), unused, dtype=np.uint16)
    for row_index, action_id in enumerate(np.asarray(packed_ids, dtype=np.int64).tolist()):
        decoded = action_catalog.decode(int(action_id))
        rows[row_index, 0] = np.uint16(family_index[decoded.family])
        if decoded.hand_index is not None:
            rows[row_index, 1] = np.uint16(decoded.hand_index)
        if decoded.stage_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.stage_slot)
        if decoded.from_slot is not None:
            rows[row_index, 1] = np.uint16(decoded.from_slot)
        if decoded.to_slot is not None:
            rows[row_index, 2] = np.uint16(decoded.to_slot)
        if decoded.slot is not None:
            rows[row_index, 1] = np.uint16(decoded.slot)
        if decoded.attack_type is not None:
            rows[row_index, 2] = np.uint16(attack_type_index[decoded.attack_type])
        if decoded.index is not None:
            rows[row_index, 1] = np.uint16(decoded.index)
    return rows


def _structured_metric_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 26,
                "pass_action_id": 25,
                "constants": [["MAX_HAND", 1], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 5},
                    {"name": "main_move", "base": 5, "count": 20},
                    {"name": "pass", "base": 25, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0]],
            }
        }
    )


def _teacher_aux_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 20,
                "pass_action_id": 19,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 3},
                    {"name": "main_move", "base": 13, "count": 6},
                    {"name": "pass", "base": 19, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def _mulligan_metric_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 9,
                "pass_action_id": 8,
                "constants": [["MAX_HAND", 4], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "mulligan_confirm", "base": 0, "count": 1},
                    {"name": "mulligan_select", "base": 1, "count": 4},
                    {"name": "attack", "base": 5, "count": 3},
                    {"name": "pass", "base": 8, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_impala_learner_writes_checkpoint_metadata_using_update_count(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
    )

    for _ in range(4):
        result = learner.update({})
        assert result["loss"] == 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_metadata_2.json").is_file()
    assert (checkpoint_dir / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_checkpoint_metadata_records_scope_update_and_policy_version(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=3,
    )

    for _ in range(3):
        learner.update({})

    checkpoint_metadata = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_metadata_3.json").read_text(encoding="utf-8")
    )
    assert checkpoint_metadata == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "policy_version": 1,
        "update_count": 3,
    }


def test_impala_learner_writes_fault_bundle_on_nonfinite_forward_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNLogitModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner forward_logits; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "forward_logits"
    assert payload["context"]["forward_logits_nonfinite_indices"]["data"] == [[0, 0, 0], [1, 0, 0]]


def test_impala_learner_writes_fault_bundle_on_nonfinite_gradients(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNGradientModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner gradients; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "gradients"
    assert "logit_bias" in payload["context"]["bad_gradient_names"]


def test_impala_learner_packed_legal_actions_match_dense_mask_loss() -> None:
    torch.manual_seed(0)
    dense_model = TinyPolicyValueModel()
    packed_model = TinyPolicyValueModel()
    packed_model.load_state_dict(dense_model.state_dict())
    dense_learner = ImpalaLearner(model=dense_model, pass_action_id=2)
    packed_learner = ImpalaLearner(model=packed_model, pass_action_id=2)

    legal_mask = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    actions = np.asarray([[0], [2]], dtype=np.int64)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": actions,
        "legal_mask": legal_mask,
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask)
    packed_batch = dict(batch)
    packed_batch["legal_actions"] = LegalActionBatch.from_packed(packed_ids, packed_offsets)
    packed_batch["legal_mask"] = None

    dense_loss, dense_metrics = dense_learner._loss_and_metrics(batch)
    packed_loss, packed_metrics = packed_learner._loss_and_metrics(packed_batch)

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_batch["legal_mask"] is None
    assert dense_metrics == pytest.approx(packed_metrics)


def test_summarize_structured_policy_metrics_reports_mainmove_pressure() -> None:
    action_catalog = _structured_metric_catalog()
    main_move_02_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (
            action_catalog.decode(action_id).family == "main_move"
            and action_catalog.decode(action_id).from_slot == 0
            and action_catalog.decode(action_id).to_slot == 2
        )
    )
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)

    legal_mask[0, 0, [0, main_move_02_action, 25]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, main_move_02_action] = 2.0
    logits[0, 0, 25] = 1.0

    legal_mask[1, 0, [0, main_move_02_action, 25]] = True
    logits[1, 0, 0] = 3.0
    logits[1, 0, main_move_02_action] = 0.0
    logits[1, 0, 25] = 1.0

    metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)

    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(0.5)
    assert 0.0 < metrics["structured_main_move_share_when_play_available"] < 1.0
    assert (
        metrics["structured_main_play_character_mass"]
        + metrics["structured_main_move_mass"]
        + metrics["structured_pass_mass"]
    ) == pytest.approx(1.0)
    assert 0.0 < metrics["structured_exact_action_concentration"] <= 1.0


def test_summarize_structured_policy_metrics_matches_packed_meta_path() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    packed_metrics = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_structured_teacher_auxiliary_metrics_supervises_slot_groups_not_hand_indices() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Row 0: two different hand indices map to the same play slot. Slot supervision should
    # treat their combined probability mass as correct.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1: attack family with the correct attack type.
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        teacher_slot=torch.tensor([[0], [0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [11]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.1,
        attack_type_coef=0.05,
        action_coef=0.15,
        same_family_action_coef=0.2,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_loss"] < 0.05
    assert metrics["teacher_action_loss"] < 0.35
    assert metrics["teacher_same_family_action_loss"] < 0.35


def test_compute_structured_teacher_auxiliary_metrics_groups_main_move_targets_by_destination_slot() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_actions_by_target: dict[int, list[int]] = {}
    for action_id in range(action_catalog.action_space_size):
        decoded = action_catalog.decode(action_id)
        if decoded.family != "main_move" or decoded.to_slot is None:
            continue
        move_actions_by_target.setdefault(int(decoded.to_slot), []).append(int(action_id))
    target_slot, target_actions = next(
        (slot, action_ids) for slot, action_ids in move_actions_by_target.items() if len(action_ids) >= 2
    )
    preferred_move, alternate_move = target_actions[:2]

    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [preferred_move, alternate_move, action_catalog.pass_action_id]] = True
    logits[0, 0, preferred_move] = 1.0
    logits[0, 0, alternate_move] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -4.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        "teacher_slot": torch.tensor([[target_slot]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[preferred_move]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 1.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 1.0,
    }

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)
    assert dense_metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert dense_metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert dense_metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)

    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
    }

    logits[0, 0, 0] = 4.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -5.0
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    logits[0, 0, 0] = 0.5
    logits[0, 0, 5] = 4.0
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    teacher_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    misaligned_view = _packed_structured_legal_view(
        logits=torch.tensor([4.0, 0.5, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=misaligned_view,
        **cast(Any, teacher_kwargs),
    )

    aligned_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=aligned_view,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_gates_public_heuristic_by_family() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    common_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "packed_view": packed_view,
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    allowed_loss, allowed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("main_play_character",),
        **cast(Any, common_kwargs),
    )
    gated_loss, gated_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("attack",),
        **cast(Any, common_kwargs),
    )

    assert float(allowed_loss.detach()) > 0.0
    assert allowed_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert allowed_metrics["teacher_public_heuristic_loss"] > 0.0
    assert float(gated_loss.detach()) == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_loss"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_matches_packed_meta_path() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0
    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[0, 0, move_action] = 4.0
    logits[0, 0, competing_move_action] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_skips_unsupported_packed_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1 carries attack teacher labels but only exposes pass legally, which previously
    # produced NaNs in the packed grouped-log-prob path.
    legal_mask[1, 0, [19]] = True
    logits[1, 0, 19] = 1.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    packed_loss, packed_metrics, packed_context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    assert torch.isfinite(packed_loss)
    assert np.isfinite(packed_metrics["teacher_aux_loss"])
    assert np.isfinite(packed_metrics["teacher_family_loss"])
    assert np.isfinite(packed_metrics["teacher_slot_loss"])
    assert np.isfinite(packed_metrics["teacher_attack_type_loss"])
    assert np.isfinite(packed_metrics["teacher_action_loss"])
    assert np.isfinite(packed_metrics["teacher_same_family_action_loss"])
    assert packed_metrics["teacher_action_supported_fraction"] == pytest.approx(0.5)
    assert packed_metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(0.5)
    assert "teacher_attack_type_log_probs" not in packed_context
    assert "teacher_family_log_probs" in packed_context
    assert "teacher_action_log_probs" in packed_context
    assert "teacher_same_family_action_log_probs" in packed_context
    assert not torch.isnan(packed_context["teacher_family_log_probs"]).any()


def test_compute_structured_teacher_auxiliary_metrics_reports_within_family_tactical_miss() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Two play-character actions share the same play slot. The model picks the wrong hand index,
    # so family and slot stay correct while the exact within-family choice is wrong.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 1.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, 19] = -4.0

    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
    )

    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_same_family_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["main_move"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_action_logp=torch.tensor([[-0.1], [-0.2]], dtype=torch.float32),
        factorized_same_family_top_action_ids=torch.tensor([[0], [move_action]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(1.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_exact_action_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_top_action_ids=torch.tensor([[0]], dtype=torch.long),
        factorized_same_family_action_logp=torch.tensor([[-0.1]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_hand_targets() -> None:
    action_catalog = ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 22,
                "pass_action_id": 21,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "clock_from_hand", "base": 10, "count": 2},
                    {"name": "attack", "base": 12, "count": 3},
                    {"name": "main_move", "base": 15, "count": 6},
                    {"name": "pass", "base": 21, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    play_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_play_character"
        and action_catalog.decode(action_id).hand_index is not None
    )
    clock_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "clock_from_hand"
        and action_catalog.decode(action_id).hand_index is not None
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["clock_from_hand"]] = 3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[play_action], [clock_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_arg0_logp=torch.tensor([[-0.05], [-0.10]], dtype=torch.float32),
        factorized_same_family_top_arg0=torch.tensor([[play_hand], [clock_hand]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_clock_from_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_hand_loss"] == pytest.approx(0.075)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_move_source_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_explicit_move_source_labels() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[-1]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
        teacher_move_source=torch.tensor([[int(move_decoded.from_slot or 0)]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_reports_family_coverage_on_active_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.zeros((4, 1, len(action_catalog.families)), dtype=torch.float32)
    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [
                [family_index["main_play_character"]],
                [family_index["main_move"]],
                [family_index["attack"]],
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [1], [0], [2]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1], [0], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [5], [11], [5]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True], [True], [False]], dtype=torch.bool),
        loss_mask=torch.tensor([[1.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )

    assert metrics["teacher_active_fraction"] == pytest.approx(0.75)
    assert metrics["teacher_main_play_character_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_main_move_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_attack_fraction"] == pytest.approx(0.0)


def test_impala_learner_auxiliary_update_uses_factorized_same_family_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_same_family_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_hand_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_hand_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)


def test_impala_learner_factorized_policy_anchor_penalizes_post_anchor_drift() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(
        model=model,
        policy_anchor_coef=0.5,
        policy_anchor_temperature=1.0,
    )
    learner._ensure_policy_anchor_model()
    with torch.no_grad():
        model.bias.fill_(2.0)
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["policy_anchor_coef_active"] == pytest.approx(0.5)
    assert metrics["policy_anchor_loss"] > 0.0
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(metrics["policy_anchor_loss"] * 0.5)
    assert metrics["policy_anchor_candidate_count"] == pytest.approx(float(packed_ids.shape[0]))
    assert model.factorized_candidate_logp_calls == 1
    assert learner._policy_anchor_model is not None


def test_impala_learner_reset_policy_anchor_refreshes_current_weights() -> None:
    model = TinyPolicyValueModel()
    learner = ImpalaLearner(model=model, policy_anchor_coef=0.5)
    learner._ensure_policy_anchor_model()

    with torch.no_grad():
        model.policy.bias.fill_(3.0)
    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is not None
    anchor_bias = dict(learner._policy_anchor_model.state_dict())["policy.bias"]
    assert torch.equal(anchor_bias, model.policy.bias.detach())


def test_impala_learner_reset_policy_anchor_clears_disabled_anchor() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    learner._ensure_policy_anchor_model()

    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is None


def test_impala_learner_auxiliary_update_uses_factorized_teacher_action_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_move_source_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_move_source_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([move_action, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_move"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[int(move_decoded.to_slot or 0)]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[move_action]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_public_heuristic_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.trunk_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_loss"] > 0.0
    assert metrics["teacher_public_heuristic_top1_mass"] < 0.1


def test_impala_learner_factorized_margin_aux_uses_factorized_candidate_log_probs_without_public_teacher() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_margin_coef=1.0,
        teacher_action_margin=0.5,
        teacher_same_family_action_margin_coef=1.0,
        teacher_same_family_action_margin=0.5,
        teacher_public_heuristic_coef=0.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 0
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_action_margin_mean"] > 0.5
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_margin_mean"] > 0.5


def test_impala_learner_auxiliary_update_averages_multiple_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 3
    assert model.public_target_profiles == ["base", "aggressive", "control"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_auxiliary_update_cycles_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["aggressive"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_public_heuristic_profiles_fall_back_to_base_after_end_update() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=0,
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["base"]


def test_impala_learner_mixed_precision_flag_disables_amp_on_cpu() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), mixed_precision=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["loss"] != 0.0
    assert learner._amp_enabled is False
    assert learner._grad_scaler is None


def test_impala_learner_uses_compiled_forward_model_when_provided() -> None:
    base_model = TinyPolicyValueModel(action_dim=2)
    compiled_proxy = ForwardProxyModel(base_model)
    learner = ImpalaLearner(model=base_model, compiled_model=compiled_proxy)

    loss, _metrics = learner._loss_and_metrics(_simple_training_batch())

    assert float(loss.detach()) != 0.0
    assert compiled_proxy.forward_calls == 2


def test_impala_learner_dense_trajectory_retention_is_separate_from_policy_train_mask() -> None:
    torch.manual_seed(0)
    base_model = TinyPolicyValueModel(action_dim=2)
    retention_model = TinyPolicyValueModel(action_dim=2)
    retention_model.load_state_dict(base_model.state_dict())
    base_learner = ImpalaLearner(model=base_model)
    retention_learner = ImpalaLearner(model=retention_model, trajectory_retention_coef=0.4)
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    base_loss, _base_metrics = base_learner._loss_and_metrics(batch)
    retention_loss, retention_metrics = retention_learner._loss_and_metrics(batch)

    assert retention_metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert retention_metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_supported_fraction"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_weighted_loss"] > 0.0
    assert float(retention_loss.detach()) == pytest.approx(
        float(base_loss.detach()) + retention_metrics["trajectory_retention_weighted_loss"]
    )


def test_impala_learner_forward_time_major_matches_manual_legacy_rollout() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=3)
    learner = ImpalaLearner(model=model)
    obs = torch.tensor(
        [
            [[0.25, -0.5], [1.0, 0.0]],
            [[-0.75, 0.5], [0.125, 0.25]],
        ],
        dtype=torch.float32,
    )
    initial_hidden = torch.ones((2, 1), dtype=torch.float32)

    with torch.no_grad():
        learner_logits, learner_values = learner._forward_time_major(obs, initial_hidden_state=initial_hidden)

        manual_hidden = initial_hidden
        manual_logits_steps: list[torch.Tensor] = []
        manual_value_steps: list[torch.Tensor] = []
        for step_obs in obs.unbind(dim=0):
            step_logits, step_value, manual_hidden = model(step_obs, manual_hidden)
            manual_logits_steps.append(step_logits)
            manual_value_steps.append(step_value)

    torch.testing.assert_close(learner_logits, torch.stack(manual_logits_steps, dim=0))
    torch.testing.assert_close(learner_values, torch.stack(manual_value_steps, dim=0))


def test_impala_learner_auxiliary_update_optimizes_teacher_only_loss() -> None:
    torch.manual_seed(0)

    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((2, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    legal_mask[1, 0, [10, 11, 12, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_auxiliary_update_handles_batches_without_valid_teacher_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((1, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[False]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_raw_vtrace_inputs_use_current_policy_logp_for_importance_weights() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": (action_logp - 2.0).cpu().numpy().astype(np.float32),
        "behavior_values": values.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, metrics = learner._loss_and_metrics(raw_batch)

    assert metrics["vtrace_rho_p50"] > 7.0
    assert metrics["vtrace_rho_p95"] > 7.0
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(1.0)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)


def test_impala_learner_raw_vtrace_uses_behavior_logp_on_non_train_rows_dense() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=10.0, vtrace_c_bar=10.0)
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, _values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )
    behavior_logp = action_logp.clone()
    behavior_logp[1, 0] = behavior_logp[1, 0] - 3.0

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    torch.testing.assert_close(context["vtrace_rhos"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(context["vtrace_rhos"][1, 0], torch.tensor(1.0))
    assert context["policy_train_mask"].tolist() == [[1.0], [0.0]]


def test_impala_learner_trains_value_on_non_policy_rows_by_default() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_train_fraction"] == pytest.approx(1.0)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert context["value_train_mask"].tolist() == [[1.0], [1.0]]


def test_impala_learner_accepts_explicit_value_train_mask() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "value_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["value_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_loss"] == pytest.approx(0.0)
    assert context["value_train_mask"].tolist() == [[1.0], [0.0]]


def test_impala_learner_raw_vtrace_inputs_use_current_learner_values_for_targets() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, -0.5]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)

    with torch.no_grad():
        forward = learner._forward_time_major(torch.from_numpy(obs))
        logits = forward.logits
        assert logits is not None
        values = forward.values
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )

    log_rho = -0.2
    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": (action_logp - log_rho).cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), 123.0, dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 2.4,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    expected_rho = float(np.exp(log_rho))
    expected_targets = values.detach() * (1.0 - expected_rho)
    assert torch.allclose(context["targets"], expected_targets, atol=1.0e-6)


def test_impala_learner_raw_vtrace_inputs_can_bootstrap_from_current_model() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=SeatAwareTinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)
    to_play_seat = np.asarray([[0]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    bootstrap_obs = np.asarray([[2.0, 0.0]], dtype=np.float32)
    bootstrap_actor = np.asarray([1], dtype=np.int64)
    final_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            to_play_seat=to_play_seat,
            initial_hidden_state=initial_hidden_state,
        )
        logits = forward.logits
        assert logits is not None
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )
        model = cast(Any, learner.model)
        expected_bootstrap = model.value_seat_aware(
            torch.from_numpy(bootstrap_obs),
            torch.from_numpy(bootstrap_actor),
            torch.from_numpy(final_hidden_state),
        )

    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": action_logp.cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), -77.0, dtype=np.float32),
        "bootstrap_value": np.full((1,), 123.0, dtype=np.float32),
        "bootstrap_obs": bootstrap_obs,
        "bootstrap_actor": bootstrap_actor,
        "final_hidden_state": final_hidden_state,
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    assert torch.allclose(context["targets"], expected_bootstrap.reshape(1, 1), atol=1.0e-6)


def test_impala_learner_forward_time_major_requires_packed_meta_for_structured_updates() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    legal_actions = LegalActionBatch.from_packed(
        np.asarray([0, 5, 19], dtype=np.uint32),
        np.asarray([0, 3], dtype=np.uint32),
        action_space=action_catalog.action_space_size,
    )

    with pytest.raises(ValueError, match="packed legal_actions metadata"):
        learner._forward_time_major(
            obs,
            to_play_seat=np.asarray([[0]], dtype=np.int64),
            legal_actions=legal_actions,
        )


def test_impala_learner_forward_time_major_uses_sequence_fast_path_and_records_packed_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=SequenceStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, SequenceStructuredTeacherModel)
    assert logits.shape == (2, 1, action_catalog.action_space_size)
    assert values.shape == (2, 1)
    assert model.sequence_calls == 1
    assert model.step_calls == 0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(6.0)
    assert learner._active_timing_metrics["packed_candidate_rows"] == pytest.approx(2.0)
    assert learner._active_timing_metrics["avg_legal_actions_per_row"] == pytest.approx(3.0)
    assert learner._active_timing_metrics["timer_learner_forward_time_major_ms"] >= 0.0


def test_impala_learner_forward_time_major_uses_trunk_sequence_path_and_records_breakdown() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TrunkStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    packed_logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, TrunkStructuredTeacherModel)
    assert packed_logits.shape == (6,)
    assert values.shape == (2, 1)
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.sequence_calls == 0
    assert learner._active_timing_metrics["timer_learner_trunk_ms"] >= 0.0
    assert learner._active_timing_metrics["timer_learner_packed_scorer_ms"] >= 0.0


def test_impala_learner_uses_factorized_legal_policy_path_for_loss_and_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        profile_timers=True,
        trajectory_retention_coef=0.06,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.1], [0.2]], dtype=np.float32),
            pg_advantages=np.asarray([[1.0], [0.5]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    loss, metrics = learner._loss_and_metrics(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert float(loss.detach()) != 0.0
    assert model.factorized_calls == 1
    assert learner._active_timing_metrics["timer_learner_factorized_policy_ms"] >= 0.0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(7.0)
    assert metrics["entropy"] > 0.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(0.25)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(0.015)


def test_impala_learner_restricts_packed_policy_scoring_to_train_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[5], [11]], dtype=np.int64),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=action_catalog.action_space_size,
        ),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": np.asarray([[-2.0], [-3.0]], dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert float(loss.detach()) != 0.0
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.scorer_row_count == 1
    assert model.scorer_candidate_count == 3
    assert learner._active_timing_metrics["packed_candidate_train_rows"] == pytest.approx(1.0)
    assert learner._active_timing_metrics["packed_candidate_train_count"] == pytest.approx(3.0)
    assert float(context["vtrace_rhos"][1, 0]) == pytest.approx(1.0)
    assert float(context["vtrace_rhos"][0, 0]) > 1.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)


def test_impala_learner_packed_raw_vtrace_rho_is_one_when_behavior_matches_policy() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    obs = np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32)
    actions = np.asarray([[5], [11]], dtype=np.int64)
    to_play_seat = np.asarray([[0], [1]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            legal_actions=legal_actions,
        )
        assert forward.packed_logits is not None
        behavior_logp, _entropy = packed_scores_action_logp_and_entropy(
            forward.packed_logits,
            torch.as_tensor(packed_ids, dtype=torch.long),
            torch.as_tensor(packed_offsets, dtype=torch.long),
            torch.from_numpy(actions),
            pass_action_id=action_catalog.pass_action_id,
        )
    learner._active_timing_metrics = {}
    batch = {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    torch.testing.assert_close(context["action_logp"], behavior_logp)
    torch.testing.assert_close(context["vtrace_rhos"], torch.ones_like(context["vtrace_rhos"]))
    assert metrics["target_behavior_logp_delta_abs_p99"] == pytest.approx(0.0)
    assert metrics["target_behavior_train_logp_delta_abs_p99"] == pytest.approx(0.0)


def test_impala_learner_auxiliary_update_uses_factorized_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)


def test_impala_learner_reports_reward_advantage_and_chosen_action_metrics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), pass_action_id=1)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.25], [-0.5]], dtype=np.float32),
            pg_advantages=np.asarray([[1.5], [-0.25]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "rewards": np.asarray([[0.0], [1.0]], dtype=np.float32),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["reward_std"] == pytest.approx(0.5)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["reward_min"] == pytest.approx(0.0)
    assert metrics["reward_max"] == pytest.approx(1.0)
    assert metrics["reward_nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["reward_positive_fraction"] == pytest.approx(0.5)
    assert metrics["reward_negative_fraction"] == pytest.approx(0.0)
    assert metrics["advantage_mean"] == pytest.approx(0.625)
    assert metrics["advantage_abs_mean"] == pytest.approx(0.875)
    assert metrics["target_mean"] == pytest.approx(-0.125)
    assert metrics["target_abs_mean"] == pytest.approx(0.375)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.5)
    assert metrics["chosen_pass_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_pass_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_nonpass_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)


def test_impala_learner_reports_family_chosen_action_outcome_metrics() -> None:
    catalog = _mulligan_metric_catalog()

    metrics = _chosen_action_outcome_metrics(
        actions=torch.tensor([[0], [1], [3], [5], [8]], dtype=torch.long),
        loss_mask=torch.tensor([[True], [True], [False], [True], [True]]),
        rewards=torch.tensor([[0.0], [1.0], [99.0], [2.0], [3.0]], dtype=torch.float32),
        advantages=torch.tensor([[0.5], [-0.25], [99.0], [1.25], [-1.0]], dtype=torch.float32),
        action_catalog=catalog,
        pass_action_id=catalog.pass_action_id,
    )

    assert metrics["chosen_mulligan_confirm_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_confirm_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_mulligan_confirm_train_advantage_mean"] == pytest.approx(0.5)
    assert metrics["chosen_mulligan_select_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_select_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_mulligan_select_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_attack_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_attack_train_advantage_mean"] == pytest.approx(1.25)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_main_play_character_train_fraction"] == pytest.approx(0.0)


def test_impala_learner_amp_overflow_is_reported_without_raising() -> None:
    learner = ImpalaLearner(model=NaNGradientModel())
    cast(Any, learner)._grad_scaler = FakeGradScaler(overflow=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["amp_grad_overflow"] == 1.0
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert learner.update_count == 1
