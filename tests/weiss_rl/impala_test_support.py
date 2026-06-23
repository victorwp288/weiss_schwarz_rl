from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.impala import (
    ImpalaLearner,
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


def _teacher_aux_hand_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
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


class _FiniteRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, torch.Tensor]] = []

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        del batch, context
        self.calls.append((name, tensor))


__all__ = (
    "FactorizedStructuredTeacherModel",
    "FakeGradScaler",
    "ForwardProxyModel",
    "ImpalaLearner",
    "NaNGradientModel",
    "NaNLogitModel",
    "SeatAwareTinyPolicyValueModel",
    "SequenceStructuredTeacherModel",
    "TinyPolicyValueModel",
    "TinyStructuredTeacherModel",
    "TrunkStructuredTeacherModel",
    "_FiniteRecorder",
    "_mulligan_metric_catalog",
    "_packed_ids_from_mask",
    "_packed_meta_from_ids",
    "_simple_training_batch",
    "_structured_metric_catalog",
    "_teacher_aux_catalog",
    "_teacher_aux_hand_catalog",
)
