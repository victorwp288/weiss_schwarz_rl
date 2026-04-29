"""Structured-v2 legal action scoring head and packed-action helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from weiss_rl import structured_sampling as _structured_sampling
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.card_table import card_feature_table
from weiss_rl.eval.heuristic_public import heuristic_public_scoring_profile
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.model_layers import build_mlp_stack as _build_mlp_stack
from weiss_rl.observation_layout import ObservationSlice
from weiss_rl.structured_candidate_features import StructuredCandidateFeaturesMixin as _StructuredCandidateFeaturesMixin
from weiss_rl.structured_factorized import (
    FactorizedConditionalLogProbs as _FactorizedConditionalLogProbs,
)
from weiss_rl.structured_factorized import FactorizedEvaluationResult as _FactorizedEvaluationResult
from weiss_rl.structured_factorized import FactorizedFamilyPlan as _FactorizedFamilyPlan
from weiss_rl.structured_factorized import FactorizedLegalityPlan as _FactorizedLegalityPlan
from weiss_rl.structured_factorized import factorized_local_row_indices as _factorized_local_row_indices
from weiss_rl.structured_factorized import scatter_factorized_row_values as _scatter_factorized_row_values
from weiss_rl.structured_observation import (
    StructuredObservationContract as _StructuredObservationContract,
)
from weiss_rl.structured_observation import (
    build_structured_observation_contract as _build_structured_observation_contract,  # noqa: F401
)
from weiss_rl.structured_observation import (
    masked_max_pool as _masked_max_pool,
)
from weiss_rl.structured_observation import (
    masked_mean_pool as _masked_mean_pool,
)
from weiss_rl.structured_observation import (
    optional_embedding as _optional_embedding,
)
from weiss_rl.structured_public_heuristic import StructuredPublicHeuristicMixin as _StructuredPublicHeuristicMixin

_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS = frozenset({0, 1, 2})
_PUBLIC_HEURISTIC_BACK_ROW_SLOTS = frozenset({3, 4})
_PUBLIC_HEURISTIC_CENTER_SLOT = 1
_PUBLIC_HEURISTIC_SLOT_PREFERENCE = {
    0: 20.0,
    1: 30.0,
    2: 15.0,
    3: 8.0,
    4: 6.0,
}


def _negative_logits_fill_value(dtype: torch.dtype) -> float:
    return _structured_sampling.negative_logits_fill_value(dtype)


def _packed_row_indices(offsets: Tensor) -> Tensor:
    return _structured_sampling.packed_row_indices(offsets)


@dataclass(frozen=True, slots=True)
class _PackedScoringPlan:
    row_indices: Tensor
    family_ids: Tensor
    arg0: Tensor
    arg1: Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.family_ids.shape[0])

    def slice(self, start: int, end: int) -> _PackedScoringPlan:
        return _PackedScoringPlan(
            row_indices=self.row_indices[start:end],
            family_ids=self.family_ids[start:end],
            arg0=self.arg0[start:end],
            arg1=self.arg1[start:end],
        )


def _packed_row_log_z(scores: Tensor, offsets: Tensor) -> Tensor:
    return _structured_sampling.packed_row_log_z(scores, offsets)


def _packed_local_cdf(probabilities: Tensor, offsets: Tensor) -> Tensor:
    return _structured_sampling.packed_local_cdf(probabilities, offsets)


def _uniform_from_seeds(sample_seeds: Tensor, *, dtype: torch.dtype) -> Tensor:
    return _structured_sampling.uniform_from_seeds(sample_seeds, dtype=dtype)


def _derived_sample_seeds(sample_seeds: Tensor, *, salt: int) -> Tensor:
    return _structured_sampling.derived_sample_seeds(sample_seeds, salt=salt)


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    return _structured_sampling.masked_log_softmax(logits, mask)


def _masked_entropy_from_log_probs(log_probs: Tensor, mask: Tensor) -> Tensor:
    return _structured_sampling.masked_entropy_from_log_probs(log_probs, mask)


def _sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
) -> tuple[Tensor, Tensor]:
    return _structured_sampling.sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=sample_seeds,
        default_index=default_index,
        uniform_fn=_uniform_from_seeds,
    )


def _sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
) -> tuple[Tensor, Tensor]:
    return _structured_sampling.sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=pass_action_id,
        uniform_fn=_uniform_from_seeds,
        packed_local_cdf_fn=_packed_local_cdf,
    )


class _StructuredLegalActionHead(_StructuredCandidateFeaturesMixin, _StructuredPublicHeuristicMixin, nn.Module):
    def __init__(
        self,
        *,
        latent_width: int,
        action_catalog: ActionCatalog,
        observation_contract: _StructuredObservationContract,
        card_table: Mapping[str, Any] | None,
        action_feature_width: int,
        layer_norm: bool,
        dropout_p: float,
        candidate_scoring_chunk_size: int = 65536,
        cuda_learner_candidate_scoring_chunk_size: int = 262144,
        public_heuristic_logit_bias_scale: float = 0.0,
        public_heuristic_actor_logit_bias_scale: float = -1.0,
        public_heuristic_logit_bias_families: tuple[str, ...] = (),
        public_heuristic_logit_bias_profile: str = "base",
    ) -> None:
        super().__init__()
        if latent_width <= 0:
            raise ValueError(f"latent_width must be >= 1, got {latent_width}")
        if action_feature_width <= 0:
            raise ValueError(f"action_feature_width must be >= 1, got {action_feature_width}")
        if candidate_scoring_chunk_size <= 0:
            raise ValueError(f"candidate_scoring_chunk_size must be >= 1, got {candidate_scoring_chunk_size}")
        if cuda_learner_candidate_scoring_chunk_size <= 0:
            raise ValueError(
                "cuda_learner_candidate_scoring_chunk_size must be >= 1, "
                f"got {cuda_learner_candidate_scoring_chunk_size}"
            )
        if public_heuristic_logit_bias_scale < 0.0:
            raise ValueError(
                f"public_heuristic_logit_bias_scale must be >= 0.0, got {public_heuristic_logit_bias_scale}"
            )
        if public_heuristic_actor_logit_bias_scale < 0.0 and public_heuristic_actor_logit_bias_scale != -1.0:
            raise ValueError(
                "public_heuristic_actor_logit_bias_scale must be >= 0.0 or -1.0, "
                f"got {public_heuristic_actor_logit_bias_scale}"
            )
        self.action_dim = int(action_catalog.action_space_size)
        self._stage_slot_count = max(int(action_catalog.max_stage), 1)
        self._observation_contract = observation_contract
        self._card_vocab_size = 32768
        self._public_heuristic_logit_bias_scale = float(public_heuristic_logit_bias_scale)
        self._public_heuristic_actor_logit_bias_scale = float(
            public_heuristic_logit_bias_scale
            if public_heuristic_actor_logit_bias_scale < 0.0
            else public_heuristic_actor_logit_bias_scale
        )
        self._public_heuristic_logit_bias_profile = heuristic_public_scoring_profile(
            public_heuristic_logit_bias_profile
        )

        family_names = tuple(family.name for family in action_catalog.families)
        family_index = {name: index for index, name in enumerate(family_names)}
        unknown_public_bias_families = sorted(
            {name for name in public_heuristic_logit_bias_families if name not in family_index}
        )
        if unknown_public_bias_families:
            raise ValueError(
                "public_heuristic_logit_bias_families contains unknown action families: "
                + ", ".join(unknown_public_bias_families)
            )
        attack_type_names = tuple(action_catalog.attack_type_names)
        attack_type_index = {name: index for index, name in enumerate(attack_type_names)}
        self._meta_unused = int(np.iinfo(np.uint16).max)
        self._attack_family_id = int(family_index.get("attack", -1))
        self._encore_pay_family_id = int(family_index.get("encore_pay", -1))
        self._encore_decline_family_id = int(family_index.get("encore_decline", -1))
        self._play_character_family_id = int(family_index.get("main_play_character", -1))
        self._main_event_family_id = int(family_index.get("main_play_event", -1))
        self._clock_from_hand_family_id = int(family_index.get("clock_from_hand", -1))
        self._climax_play_family_id = int(family_index.get("climax_play", -1))
        self._mulligan_select_family_id = int(family_index.get("mulligan_select", -1))
        self._mulligan_confirm_family_id = int(family_index.get("mulligan_confirm", -1))
        self._main_move_family_id = int(family_index.get("main_move", -1))
        self._choice_select_family_id = int(family_index.get("choice_select", -1))
        self.register_buffer(
            "_public_heuristic_bias_family_ids",
            torch.as_tensor(
                tuple(int(family_index[name]) for name in public_heuristic_logit_bias_families),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self._next_page_family_id = int(family_index.get("choice_next_page", -1))
        self._prev_page_family_id = int(family_index.get("choice_prev_page", -1))
        self._level_up_family_id = int(family_index.get("level_up", -1))
        self._trigger_order_family_id = int(family_index.get("trigger_order", -1))
        self._pass_family_id = int(family_index.get("pass", -1))
        self._frontal_attack_type_id = int(attack_type_index.get("frontal", -1))
        self._side_attack_type_id = int(attack_type_index.get("side", -1))
        self._direct_attack_type_id = int(attack_type_index.get("direct", -1))
        self._hand_family_ids = tuple(
            family_id
            for family_id in (
                self._main_event_family_id,
                self._clock_from_hand_family_id,
                self._climax_play_family_id,
                self._mulligan_select_family_id,
            )
            if family_id >= 0
        )

        family_ids = np.zeros((self.action_dim,), dtype=np.int64)
        action_arg0 = np.full((self.action_dim,), -1, dtype=np.int64)
        action_arg1 = np.full((self.action_dim,), -1, dtype=np.int64)
        hand_indices = np.full((self.action_dim,), -1, dtype=np.int64)
        stage_slots = np.full((self.action_dim,), -1, dtype=np.int64)
        from_slots = np.full((self.action_dim,), -1, dtype=np.int64)
        to_slots = np.full((self.action_dim,), -1, dtype=np.int64)
        attack_slots = np.full((self.action_dim,), -1, dtype=np.int64)
        attack_types = np.full((self.action_dim,), -1, dtype=np.int64)
        generic_indices = np.full((self.action_dim,), -1, dtype=np.int64)
        for action_id in range(self.action_dim):
            decoded = action_catalog.decode(action_id)
            family_ids[action_id] = family_index.get(decoded.family, 0)
            if decoded.hand_index is not None:
                action_arg0[action_id] = int(decoded.hand_index)
                hand_indices[action_id] = int(decoded.hand_index)
            if decoded.stage_slot is not None:
                action_arg1[action_id] = int(decoded.stage_slot)
                stage_slots[action_id] = int(decoded.stage_slot)
            if decoded.from_slot is not None:
                action_arg0[action_id] = int(decoded.from_slot)
                from_slots[action_id] = int(decoded.from_slot)
            if decoded.to_slot is not None:
                action_arg1[action_id] = int(decoded.to_slot)
                to_slots[action_id] = int(decoded.to_slot)
            if decoded.slot is not None:
                action_arg0[action_id] = int(decoded.slot)
                attack_slots[action_id] = int(decoded.slot)
            if decoded.attack_type is not None:
                action_arg1[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
                attack_types[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
            if decoded.index is not None:
                action_arg0[action_id] = int(decoded.index)
                generic_indices[action_id] = int(decoded.index)

        family_embed_dim = max(12, min(48, action_feature_width // 3))
        slot_embed_dim = max(8, min(24, action_feature_width // 5))
        card_embed_dim = max(16, min(64, action_feature_width // 2))
        slot_context_dim = max(24, action_feature_width // 2)
        state_width = max(32, int(action_feature_width))
        self._slot_context_dim = slot_context_dim

        self.family_embedding = nn.Embedding(max(len(family_names), 1), family_embed_dim)
        self.slot_embedding = nn.Embedding(self._stage_slot_count + 1, slot_embed_dim)
        self.attack_type_embedding = nn.Embedding(len(attack_type_names) + 1, slot_embed_dim)
        self.card_embedding = nn.Embedding(self._card_vocab_size, card_embed_dim)
        self.hand_position_embedding = nn.Embedding(max(int(action_catalog.max_hand), 1) + 1, card_embed_dim)
        static_feature_table = card_feature_table(card_table=card_table, vocab_size=self._card_vocab_size)
        self.register_buffer(
            "_card_static_features",
            torch.as_tensor(static_feature_table, dtype=torch.float32),
            persistent=False,
        )
        self.card_feature_projection = (
            None
            if static_feature_table.shape[1] == 0
            else _build_mlp_stack(
                input_dim=int(static_feature_table.shape[1]),
                width=card_embed_dim,
                layers=1,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
        )
        self.hand_summary_projection = _build_mlp_stack(
            input_dim=card_embed_dim * 2 + 1,
            width=slot_context_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.slot_encoder = _build_mlp_stack(
            input_dim=card_embed_dim + 7,
            width=slot_context_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.state_projection = _build_mlp_stack(
            input_dim=latent_width + slot_context_dim * 3,
            width=state_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self._family_feature_offset = 0
        self._hand_card_feature_offset = self._family_feature_offset + family_embed_dim
        self._stage_slot_feature_offset = self._hand_card_feature_offset + card_embed_dim
        self._from_slot_feature_offset = self._stage_slot_feature_offset + slot_embed_dim
        self._to_slot_feature_offset = self._from_slot_feature_offset + slot_embed_dim
        self._attack_slot_feature_offset = self._to_slot_feature_offset + slot_embed_dim
        self._attack_type_feature_offset = self._attack_slot_feature_offset + slot_embed_dim
        self._play_target_context_offset = self._attack_type_feature_offset + slot_embed_dim
        self._move_source_context_offset = self._play_target_context_offset + slot_context_dim
        self._move_target_context_offset = self._move_source_context_offset + slot_context_dim
        self._attack_source_context_offset = self._move_target_context_offset + slot_context_dim
        self._defender_context_offset = self._attack_source_context_offset + slot_context_dim
        self._numeric_feature_offset = self._defender_context_offset + slot_context_dim
        candidate_input_dim = family_embed_dim + card_embed_dim + slot_embed_dim * 5 + slot_context_dim * 5 + 11
        self._candidate_input_dim = int(candidate_input_dim)
        self.candidate_projection = _build_mlp_stack(
            input_dim=candidate_input_dim,
            width=state_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        scorer_layers: list[nn.Module] = [nn.Linear(state_width * 2, state_width)]
        if layer_norm:
            scorer_layers.append(nn.LayerNorm(state_width))
        scorer_layers.append(nn.ReLU())
        if dropout_p > 0.0:
            scorer_layers.append(nn.Dropout(p=dropout_p))
        scorer_layers.append(nn.Linear(state_width, 1))
        self.joint_scorer = nn.Sequential(*scorer_layers)
        self.family_bias = nn.Parameter(torch.zeros(max(len(family_names), 1)))
        self._candidate_scoring_chunk_size = int(candidate_scoring_chunk_size)
        self._cuda_learner_candidate_scoring_chunk_size = int(cuda_learner_candidate_scoring_chunk_size)
        self.register_buffer("_family_ids", torch.as_tensor(family_ids, dtype=torch.long))
        self.register_buffer("_action_arg0", torch.as_tensor(action_arg0, dtype=torch.long))
        self.register_buffer("_action_arg1", torch.as_tensor(action_arg1, dtype=torch.long))
        self.register_buffer("_hand_indices", torch.as_tensor(hand_indices, dtype=torch.long))
        self.register_buffer("_stage_slots", torch.as_tensor(stage_slots, dtype=torch.long))
        self.register_buffer("_from_slots", torch.as_tensor(from_slots, dtype=torch.long))
        self.register_buffer("_to_slots", torch.as_tensor(to_slots, dtype=torch.long))
        self.register_buffer("_attack_slots", torch.as_tensor(attack_slots, dtype=torch.long))
        self.register_buffer("_attack_types", torch.as_tensor(attack_types, dtype=torch.long))
        self.register_buffer("_generic_indices", torch.as_tensor(generic_indices, dtype=torch.long))

        family_count = max(len(family_names), 1)
        family_arg_kind = np.zeros((family_count,), dtype=np.int64)
        hand_family_names = {
            "mulligan_select",
            "clock_from_hand",
            "main_play_event",
            "climax_play",
        }
        slot_family_names = {"encore_pay", "encore_decline"}
        index_family_names = {"level_up", "trigger_order", "choice_select"}
        for family_name, family_id in family_index.items():
            if family_name in hand_family_names:
                family_arg_kind[family_id] = 1
            elif family_name == "main_play_character":
                family_arg_kind[family_id] = 2
            elif family_name == "main_move":
                family_arg_kind[family_id] = 3
            elif family_name == "attack":
                family_arg_kind[family_id] = 4
            elif family_name in slot_family_names:
                family_arg_kind[family_id] = 5
            elif family_name in index_family_names:
                family_arg_kind[family_id] = 6
        family_arg0_size = np.zeros((family_count,), dtype=np.int64)
        family_arg1_size = np.zeros((family_count,), dtype=np.int64)
        family_noarg_action_ids = np.full((family_count,), -1, dtype=np.int64)
        for action_id in range(self.action_dim):
            family_id = int(family_ids[action_id])
            arg0 = int(action_arg0[action_id])
            arg1 = int(action_arg1[action_id])
            if arg0 < 0 and arg1 < 0:
                family_noarg_action_ids[family_id] = action_id
                continue
            if arg0 >= 0:
                family_arg0_size[family_id] = max(family_arg0_size[family_id], arg0 + 1)
            if arg1 >= 0:
                family_arg1_size[family_id] = max(family_arg1_size[family_id], arg1 + 1)
        max_arg0 = max(int(family_arg0_size.max()) if family_arg0_size.size else 0, 1)
        max_arg1 = max(int(family_arg1_size.max()) if family_arg1_size.size else 0, 1)
        one_arg_action_ids = np.full((family_count, max_arg0), -1, dtype=np.int64)
        two_arg_action_ids = np.full((family_count, max_arg0, max_arg1), -1, dtype=np.int64)
        for action_id in range(self.action_dim):
            family_id = int(family_ids[action_id])
            arg0 = int(action_arg0[action_id])
            arg1 = int(action_arg1[action_id])
            if arg0 < 0 and arg1 < 0:
                continue
            if arg0 >= 0 and arg1 < 0:
                one_arg_action_ids[family_id, arg0] = action_id
            elif arg0 >= 0 and arg1 >= 0:
                two_arg_action_ids[family_id, arg0, arg1] = action_id
        generic_embed_dim = max(8, min(24, action_feature_width // 5))
        self.generic_index_embedding = nn.Embedding(max_arg0 + 1, generic_embed_dim)
        self.generic_candidate_projection = _build_mlp_stack(
            input_dim=generic_embed_dim,
            width=card_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.family_head = nn.Linear(state_width, family_count)
        self.hand_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim,
            width=card_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.index_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim,
            width=generic_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.slot_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim,
            width=slot_context_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.play_slot_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim + card_embed_dim,
            width=slot_context_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.move_target_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim + slot_context_dim,
            width=slot_context_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.attack_type_query_head = _build_mlp_stack(
            input_dim=state_width + family_embed_dim + slot_context_dim,
            width=slot_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.register_buffer("_family_arg_kind", torch.as_tensor(family_arg_kind, dtype=torch.long))
        self.register_buffer("_family_arg0_size", torch.as_tensor(family_arg0_size, dtype=torch.long))
        self.register_buffer("_family_arg1_size", torch.as_tensor(family_arg1_size, dtype=torch.long))
        self.register_buffer("_family_noarg_action_ids", torch.as_tensor(family_noarg_action_ids, dtype=torch.long))
        self.register_buffer("_one_arg_action_ids", torch.as_tensor(one_arg_action_ids, dtype=torch.long))
        self.register_buffer("_two_arg_action_ids", torch.as_tensor(two_arg_action_ids, dtype=torch.long))
        self._slot_family_ids = tuple(
            int(family_index[name]) for name in sorted(slot_family_names) if name in family_index
        )
        self._index_family_ids = tuple(
            int(family_index[name]) for name in sorted(index_family_names) if name in family_index
        )
        slot_preference = np.zeros((self._stage_slot_count,), dtype=np.float32)
        for slot_index in range(self._stage_slot_count):
            slot_preference[slot_index] = float(_PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(slot_index, 0.0))
        self.register_buffer(
            "_public_slot_preference", torch.as_tensor(slot_preference, dtype=torch.float32), persistent=False
        )
        self._factorized_learner_row_chunk_size = 8192
        self._factorized_actor_row_chunk_size = 32768

    def set_public_heuristic_logit_bias_scales(
        self,
        *,
        learner_scale: float | None = None,
        actor_scale: float | None = None,
    ) -> None:
        if learner_scale is not None:
            resolved = float(learner_scale)
            if resolved < 0.0:
                raise ValueError(f"public_heuristic_logit_bias_scale must be >= 0.0, got {resolved}")
            self._public_heuristic_logit_bias_scale = resolved
        if actor_scale is not None:
            resolved = float(actor_scale)
            if resolved < 0.0:
                raise ValueError(f"public_heuristic_actor_logit_bias_scale must be >= 0.0, got {resolved}")
            self._public_heuristic_actor_logit_bias_scale = resolved

    def _build_state_representation(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        observation_context: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D (batch, hidden), got shape {tuple(latent.shape)}")
        if obs.ndim != 2 or obs.shape[0] != latent.shape[0]:
            raise ValueError("structured_v2 policy head requires obs with shape (batch, observation)")
        obs_batch = obs.to(device=latent.device, dtype=torch.float32)
        resolved_context = (
            self._encode_observation_context(obs_batch) if observation_context is None else dict(observation_context)
        )
        state_repr = self.state_projection(
            torch.cat(
                [
                    latent,
                    resolved_context["hand_summary"].to(dtype=latent.dtype),
                    resolved_context["self_stage_summary"].to(dtype=latent.dtype),
                    resolved_context["opponent_stage_summary"].to(dtype=latent.dtype),
                ],
                dim=1,
            )
        )
        return state_repr, resolved_context

    def score_legal_actions(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        resolved_state_repr, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )

        masked = torch.full(
            (latent.shape[0], self.action_dim),
            _negative_logits_fill_value(latent.dtype),
            device=latent.device,
            dtype=latent.dtype,
        )
        if legal_actions is None:
            candidate_ids = torch.arange(self.action_dim, device=latent.device, dtype=torch.long)
            for row_index in range(latent.shape[0]):
                row_scores = self._score_candidates(
                    resolved_state_repr[row_index].unsqueeze(0),
                    torch.zeros((candidate_ids.shape[0],), device=latent.device, dtype=torch.long),
                    candidate_ids,
                    resolved_context,
                    scoring_mode=scoring_mode,
                )
                masked[row_index, candidate_ids] = row_scores.to(dtype=masked.dtype)
            return masked

        if legal_actions.ids is not None and legal_actions.offsets is not None:
            offsets = torch.as_tensor(legal_actions.offsets, device=latent.device, dtype=torch.long)
            if offsets.ndim != 1 or offsets.numel() != latent.shape[0] + 1:
                raise ValueError(f"packed legal offsets must have shape ({latent.shape[0] + 1},)")
            ids = torch.as_tensor(legal_actions.ids, device=latent.device, dtype=torch.long)
            if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(ids.numel()):
                raise ValueError("packed legal offsets must be a valid prefix sum")
            row_scores = self.score_packed_candidates(
                latent,
                obs=obs,
                legal_actions=legal_actions,
                observation_context=resolved_context,
                state_repr=resolved_state_repr,
                scoring_mode=scoring_mode,
            )
            if row_scores.numel() > 0:
                lengths = offsets[1:] - offsets[:-1]
                row_indices = torch.repeat_interleave(
                    torch.arange(latent.shape[0], device=latent.device, dtype=torch.long),
                    lengths,
                )
                masked[row_indices, ids] = row_scores.to(dtype=masked.dtype)
            return masked

        if legal_actions.mask is None:
            raise ValueError("legal_actions must contain either packed ids or a mask")
        legal_mask = torch.as_tensor(legal_actions.mask, device=latent.device, dtype=torch.bool)
        if legal_mask.ndim == 3 and legal_mask.shape[0] == 1:
            legal_mask = legal_mask[0]
        if legal_mask.ndim != 2 or legal_mask.shape[0] != latent.shape[0] or legal_mask.shape[1] != self.action_dim:
            raise ValueError("legal mask must have shape (batch, action) or (1, batch, action)")
        row_indices, candidate_ids = torch.nonzero(legal_mask, as_tuple=True)
        if candidate_ids.numel() > 0:
            row_scores = self._score_candidates_chunked(
                resolved_state_repr,
                row_indices.to(dtype=torch.long),
                candidate_ids.to(dtype=torch.long),
                resolved_context,
            )
            masked[row_indices, candidate_ids] = row_scores.to(dtype=masked.dtype)
        return masked

    def score_packed_candidates(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("score_packed_candidates requires packed legal ids and offsets")
        resolved_state_repr, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )
        ids = torch.as_tensor(legal_actions.ids, device=latent.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=latent.device, dtype=torch.long)
        meta = (
            None
            if legal_actions.meta is None
            else torch.as_tensor(legal_actions.meta, device=latent.device, dtype=torch.long)
        )
        if offsets.ndim != 1 or offsets.numel() != latent.shape[0] + 1:
            raise ValueError(f"packed legal offsets must have shape ({latent.shape[0] + 1},)")
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(ids.numel()):
            raise ValueError("packed legal offsets must be a valid prefix sum")
        if ids.numel() == 0:
            return latent.new_zeros((0,))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=ids,
            offsets=offsets,
            candidate_meta=meta,
        )
        return self._score_packed_candidates_chunked(
            resolved_state_repr,
            scoring_plan,
            resolved_context,
            scoring_mode=scoring_mode,
        )

    def score_packed_public_heuristic_candidates(
        self,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None or legal_actions.meta is None:
            raise ValueError(
                "score_packed_public_heuristic_candidates requires packed legal ids, offsets, and metadata"
            )
        obs_batch = torch.as_tensor(obs)
        if obs_batch.ndim != 2:
            raise ValueError("score_packed_public_heuristic_candidates expects obs to be 2D (rows, observation)")
        resolved_profile = heuristic_public_scoring_profile(scoring_profile)
        resolved_context = (
            dict(observation_context)
            if observation_context is not None
            else self._encode_observation_context(obs_batch)
        )
        ids = torch.as_tensor(legal_actions.ids, device=obs_batch.device, dtype=torch.long)
        offsets = torch.as_tensor(legal_actions.offsets, device=obs_batch.device, dtype=torch.long)
        meta = torch.as_tensor(legal_actions.meta, device=obs_batch.device, dtype=torch.long)
        if offsets.ndim != 1 or offsets.numel() != obs_batch.shape[0] + 1:
            raise ValueError(f"packed legal offsets must have shape ({obs_batch.shape[0] + 1},)")
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(ids.numel()):
            raise ValueError("packed legal offsets must be a valid prefix sum")
        if ids.numel() == 0:
            return obs_batch.new_zeros((0,))
        scoring_plan = self._build_packed_scoring_plan(
            candidate_ids=ids,
            offsets=offsets,
            candidate_meta=meta,
        )
        return self._score_packed_public_heuristic_chunked(
            scoring_plan,
            resolved_context,
            dtype=obs_batch.dtype,
            scoring_profile=resolved_profile,
        )

    def forward(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        return self.score_legal_actions(
            latent,
            obs=obs,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )

    def _encode_observation_context(self, obs_batch: Tensor) -> dict[str, Tensor]:
        batch_size = obs_batch.shape[0]
        dtype = obs_batch.dtype

        hand_ids = self._extract_card_vector(obs_batch, self._observation_contract.self_hand)
        if hand_ids.shape[1] == 0:
            hand_summary = obs_batch.new_zeros((batch_size, self._slot_context_dim))
        else:
            hand_mask = hand_ids > max(self._observation_contract.sentinel_empty_card, 0)
            hand_embeddings = self._card_representation(hand_ids, dtype=dtype)
            hand_summary = self.hand_summary_projection(
                torch.cat(
                    [
                        _masked_mean_pool(hand_embeddings, hand_mask),
                        _masked_max_pool(hand_embeddings, hand_mask),
                        hand_mask.to(dtype=dtype).mean(dim=1, keepdim=True),
                    ],
                    dim=1,
                )
            )

        self_stage_ctx, self_stage_numeric = self._encode_stage_slice(obs_batch, self._observation_contract.self_stage)
        opponent_stage_ctx, opponent_stage_numeric = self._encode_stage_slice(
            obs_batch,
            self._observation_contract.opponent_stage,
        )
        return {
            "hand_ids": hand_ids,
            "hand_summary": hand_summary,
            "self_stage_context": self_stage_ctx,
            "self_stage_numeric": self_stage_numeric,
            "self_stage_summary": self_stage_ctx.mean(dim=1),
            "self_level_count": self._extract_scalar_feature(obs_batch, self._observation_contract.self_level_count),
            "self_clock_count": self._extract_scalar_feature(obs_batch, self._observation_contract.self_clock_count),
            "opponent_stage_context": opponent_stage_ctx,
            "opponent_stage_numeric": opponent_stage_numeric,
            "opponent_stage_summary": opponent_stage_ctx.mean(dim=1),
            "choice_page_start": self._extract_header_scalar(
                obs_batch, self._observation_contract.choice_page_start_index
            ),
            "choice_total": self._extract_header_scalar(obs_batch, self._observation_contract.choice_total_index),
        }

    def _extract_scalar_feature(
        self,
        obs_batch: Tensor,
        slice_spec: ObservationSlice | None,
    ) -> Tensor:
        batch_size = obs_batch.shape[0]
        if slice_spec is None or slice_spec.length <= 0:
            return obs_batch.new_zeros((batch_size,))
        return obs_batch[:, slice_spec.start].reshape(batch_size)

    def _extract_header_scalar(
        self,
        obs_batch: Tensor,
        index: int | None,
    ) -> Tensor:
        batch_size = obs_batch.shape[0]
        if index is None:
            return obs_batch.new_zeros((batch_size,))
        return obs_batch[:, int(index)].reshape(batch_size)

    def _encode_stage_slice(
        self,
        obs_batch: Tensor,
        stage_slice: ObservationSlice | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = obs_batch.shape[0]
        dtype = obs_batch.dtype
        if stage_slice is None:
            zeros_context = obs_batch.new_zeros((batch_size, self._stage_slot_count, self._slot_context_dim))
            zeros_numeric = obs_batch.new_zeros((batch_size, self._stage_slot_count, 7))
            return zeros_context, zeros_numeric

        slot_width = max(stage_slice.length // self._stage_slot_count, 1)
        stage_values = obs_batch[:, stage_slice.start : stage_slice.stop].reshape(
            batch_size, self._stage_slot_count, slot_width
        )
        card_ids = stage_values[..., 0].to(dtype=torch.long)
        occupied = (card_ids > max(self._observation_contract.sentinel_empty_card, 0)).to(dtype=dtype)
        numeric = torch.stack(
            [
                occupied,
                self._slot_component(stage_values, 1) / 8.0,
                self._slot_component(stage_values, 2),
                self._slot_component(stage_values, 3) / 20000.0,
                self._slot_component(stage_values, 4) / 4.0,
                self._slot_component(stage_values, 5) / 4.0,
                self._slot_component(stage_values, 6),
            ],
            dim=-1,
        )
        card_embeddings = self._card_representation(card_ids, dtype=dtype)
        stage_context = self.slot_encoder(torch.cat([card_embeddings, numeric], dim=-1))
        return stage_context, numeric

    def _resolve_scoring_mode(self, scoring_mode: str) -> str:
        resolved_mode = str(scoring_mode).strip().lower()
        if resolved_mode == "auto":
            return "actor" if not torch.is_grad_enabled() else "learner"
        if resolved_mode not in {"actor", "learner"}:
            raise ValueError("scoring_mode must be one of: auto, actor, learner")
        return resolved_mode

    def _family_condition_input(self, row_states: Tensor, *, family_id: int) -> Tensor:
        family_ids = torch.full(
            (row_states.shape[0],),
            int(family_id),
            device=row_states.device,
            dtype=torch.long,
        )
        family_embed = self.family_embedding(family_ids).to(dtype=row_states.dtype)
        return torch.cat([row_states, family_embed], dim=1)

    def _factorized_row_chunk_size(self, row_states: Tensor) -> int:
        if row_states.device.type != "cuda":
            return 0
        return (
            int(self._factorized_learner_row_chunk_size)
            if torch.is_grad_enabled()
            else int(self._factorized_actor_row_chunk_size)
        )

    def _dot_product_log_probs(
        self,
        query: Tensor,
        candidate_repr: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if candidate_repr.ndim != 3 or mask.ndim != 2:
            raise ValueError("candidate_repr must be 3D and mask must be 2D")
        if candidate_repr.shape[:2] != mask.shape:
            raise ValueError("candidate_repr and mask must agree on row and candidate dimensions")
        if candidate_repr.shape[0] == 0:
            return candidate_repr.new_zeros((0, candidate_repr.shape[1]))
        logits = (candidate_repr.to(dtype=query.dtype) * query.unsqueeze(1)).sum(dim=-1)
        return _masked_log_softmax(logits, mask)

    def _build_factorized_legality_plan(
        self,
        legal_actions: LegalActionBatch,
        *,
        device: torch.device,
    ) -> _FactorizedLegalityPlan:
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("factorized structured policy requires packed legal ids and offsets")
        offsets = torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long)
        row_count = int(offsets.shape[0] - 1)
        if row_count < 0:
            raise ValueError("packed legal offsets must contain at least one row boundary")
        ids = torch.as_tensor(legal_actions.ids, device=device, dtype=torch.long)
        family_ids = self._family_ids.index_select(0, ids)
        arg0 = self._action_arg0.index_select(0, ids)
        arg1 = self._action_arg1.index_select(0, ids)
        row_indices = _packed_row_indices(offsets)
        family_count = int(self._family_arg_kind.shape[0])
        family_mask_flat = torch.zeros((row_count * family_count,), device=device, dtype=torch.bool)
        if row_indices.numel() > 0:
            family_mask_flat[row_indices * family_count + family_ids.to(dtype=torch.long)] = True
        family_mask = family_mask_flat.view(row_count, family_count)
        family_plans: dict[int, _FactorizedFamilyPlan] = {}
        for family_id in range(family_count):
            family_candidate_mask = family_ids == int(family_id)
            if not bool(family_candidate_mask.any().item()):
                continue
            family_candidate_rows = row_indices[family_candidate_mask].to(dtype=torch.long)
            family_rows = torch.unique_consecutive(family_candidate_rows)
            arg0_size = int(self._family_arg0_size[family_id].item())
            arg0_mask: Tensor | None = None
            arg1_mask: Tensor | None = None
            if arg0_size > 0:
                local_row_indices = torch.searchsorted(family_rows, family_candidate_rows)
                family_arg0 = arg0[family_candidate_mask].to(dtype=torch.long)
                arg0_mask = torch.zeros((int(family_rows.shape[0]), arg0_size), device=device, dtype=torch.bool)
                valid_arg0 = family_arg0 >= 0
                if bool(valid_arg0.any().item()):
                    arg0_mask[local_row_indices[valid_arg0], family_arg0[valid_arg0]] = True
                arg1_size = int(self._family_arg1_size[family_id].item())
                if arg1_size > 0:
                    family_arg1 = arg1[family_candidate_mask].to(dtype=torch.long)
                    valid_arg1 = valid_arg0 & (family_arg1 >= 0)
                    arg1_mask = torch.zeros(
                        (int(family_rows.shape[0]), arg0_size, arg1_size),
                        device=device,
                        dtype=torch.bool,
                    )
                    if bool(valid_arg1.any().item()):
                        flat_index = (
                            local_row_indices[valid_arg1] * (arg0_size * arg1_size)
                            + family_arg0[valid_arg1] * arg1_size
                            + family_arg1[valid_arg1]
                        )
                        arg1_mask.view(-1)[flat_index] = True
            family_plans[family_id] = _FactorizedFamilyPlan(
                row_indices=family_rows,
                arg0_mask=arg0_mask,
                arg1_mask=arg1_mask,
            )
        return _FactorizedLegalityPlan(
            row_count=row_count,
            family_mask=family_mask,
            family_plans=family_plans,
        )

    def _family_log_probs(self, row_states: Tensor, family_mask: Tensor) -> Tensor:
        family_logits = self.family_head(row_states) + self.family_bias.to(
            device=row_states.device,
            dtype=row_states.dtype,
        )
        return _masked_log_softmax(family_logits, family_mask)

    def _hand_arg0_log_probs(
        self,
        row_states: Tensor,
        *,
        family_id: int,
        hand_ids: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._hand_arg0_log_probs(
                    row_states[start:stop],
                    family_id=family_id,
                    hand_ids=hand_ids[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.hand_query_head(condition)
        hand_repr = self._card_representation(hand_ids, dtype=row_states.dtype)
        if hand_repr.shape[1] < legal_mask.shape[1]:
            raise ValueError("hand representation width must cover the factorized hand domain")
        hand_repr = hand_repr[:, : legal_mask.shape[1], :]
        return self._dot_product_log_probs(query, hand_repr, legal_mask)

    def _slot_arg0_log_probs(
        self,
        row_states: Tensor,
        *,
        family_id: int,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.slot_query_head(condition)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized slot domain")
        return self._dot_product_log_probs(query, slot_context[:, : legal_mask.shape[1], :], legal_mask)

    def _index_arg0_log_probs(
        self,
        row_states: Tensor,
        *,
        family_id: int,
        legal_mask: Tensor,
    ) -> Tensor:
        if legal_mask.shape[1] == 0:
            return row_states.new_zeros((row_states.shape[0], 0))
        condition = self._family_condition_input(row_states, family_id=family_id)
        query = self.index_query_head(condition)
        index_repr = self.generic_index_embedding(
            torch.arange(legal_mask.shape[1], device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        logits = torch.matmul(query, index_repr.transpose(0, 1))
        return _masked_log_softmax(logits, legal_mask)

    def _play_arg1_log_probs(
        self,
        row_states: Tensor,
        *,
        hand_ids: Tensor,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._play_arg1_log_probs(
                    row_states[start:stop],
                    hand_ids=hand_ids[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        hand_repr = self._card_representation(hand_ids, dtype=row_states.dtype)
        if hand_repr.shape[1] < legal_mask.shape[1]:
            raise ValueError("hand representation width must cover the factorized play domain")
        hand_repr = hand_repr[:, : legal_mask.shape[1], :]
        family_condition = self.family_embedding(
            torch.full(
                (row_states.shape[0],), self._play_character_family_id, device=row_states.device, dtype=torch.long
            )
        ).to(dtype=row_states.dtype)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        query = self.play_slot_query_head(torch.cat([state_expanded, family_expanded, hand_repr], dim=-1))
        slot_expanded = slot_context.unsqueeze(1).expand(-1, legal_mask.shape[1], -1, -1)
        logits = (slot_expanded.to(dtype=row_states.dtype) * query.unsqueeze(2)).sum(dim=-1)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)

    def _move_arg1_log_probs(
        self,
        row_states: Tensor,
        *,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._move_arg1_log_probs(
                    row_states[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        family_condition = self.family_embedding(
            torch.full((row_states.shape[0],), self._main_move_family_id, device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        source_context = slot_context.unsqueeze(1).expand(-1, legal_mask.shape[1], -1, -1)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized move domain")
        query = self.move_target_query_head(
            torch.cat([state_expanded, family_expanded, slot_context[:, : legal_mask.shape[1], :]], dim=-1)
        )
        logits = (source_context.to(dtype=row_states.dtype) * query.unsqueeze(2)).sum(dim=-1)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)

    def _attack_arg1_log_probs(
        self,
        row_states: Tensor,
        *,
        slot_context: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        chunk_size = self._factorized_row_chunk_size(row_states)
        if chunk_size > 0 and row_states.shape[0] > chunk_size:
            parts = [
                self._attack_arg1_log_probs(
                    row_states[start:stop],
                    slot_context=slot_context[start:stop],
                    legal_mask=legal_mask[start:stop],
                )
                for start in range(0, row_states.shape[0], chunk_size)
                for stop in (min(start + chunk_size, row_states.shape[0]),)
            ]
            return torch.cat(parts, dim=0)
        if legal_mask.shape[2] == 0:
            return row_states.new_zeros((row_states.shape[0], legal_mask.shape[1], 0))
        family_condition = self.family_embedding(
            torch.full((row_states.shape[0],), self._attack_family_id, device=row_states.device, dtype=torch.long)
        ).to(dtype=row_states.dtype)
        type_repr = self.attack_type_embedding(
            torch.arange(legal_mask.shape[2], device=row_states.device, dtype=torch.long) + 1
        ).to(dtype=row_states.dtype)
        family_expanded = family_condition.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        state_expanded = row_states.unsqueeze(1).expand(-1, legal_mask.shape[1], -1)
        if slot_context.shape[1] < legal_mask.shape[1]:
            raise ValueError("slot context width must cover the factorized attack domain")
        query = self.attack_type_query_head(
            torch.cat([state_expanded, family_expanded, slot_context[:, : legal_mask.shape[1], :]], dim=-1)
        )
        logits = torch.einsum("bqd,td->bqt", query, type_repr)
        return _masked_log_softmax(
            logits.reshape(-1, logits.shape[-1]), legal_mask.reshape(-1, legal_mask.shape[-1])
        ).reshape_as(logits)

    def _factorized_distributions(
        self,
        row_states: Tensor,
        *,
        legal_actions: LegalActionBatch,
        observation_context: Mapping[str, Tensor],
    ) -> tuple[
        _FactorizedLegalityPlan,
        Tensor,
        dict[int, _FactorizedConditionalLogProbs],
        dict[int, _FactorizedConditionalLogProbs],
    ]:
        plan = self._build_factorized_legality_plan(legal_actions, device=row_states.device)
        family_log_probs = self._family_log_probs(row_states, plan.family_mask)
        arg0_log_probs: dict[int, _FactorizedConditionalLogProbs] = {}
        arg1_log_probs: dict[int, _FactorizedConditionalLogProbs] = {}
        hand_ids = observation_context["hand_ids"].to(device=row_states.device, dtype=torch.long)
        self_stage_context = observation_context["self_stage_context"].to(
            device=row_states.device, dtype=row_states.dtype
        )
        for family_id, family_plan in plan.family_plans.items():
            kind = int(self._family_arg_kind[family_id].item())
            if kind == 0:
                continue
            family_rows = family_plan.row_indices
            arg0_mask = family_plan.arg0_mask
            if arg0_mask is None:
                continue
            family_row_states = row_states.index_select(0, family_rows)
            if kind in {1, 2}:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._hand_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        hand_ids=hand_ids.index_select(0, family_rows),
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            elif kind in {3, 4, 5}:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._slot_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            elif kind == 6:
                arg0_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._index_arg0_log_probs(
                        family_row_states,
                        family_id=family_id,
                        legal_mask=arg0_mask,
                    ),
                    mask=arg0_mask,
                )
            arg1_mask = family_plan.arg1_mask
            if arg1_mask is None:
                continue
            if family_id == self._play_character_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._play_arg1_log_probs(
                        family_row_states,
                        hand_ids=hand_ids.index_select(0, family_rows),
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
            elif family_id == self._main_move_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._move_arg1_log_probs(
                        family_row_states,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
            elif family_id == self._attack_family_id:
                arg1_log_probs[family_id] = _FactorizedConditionalLogProbs(
                    row_indices=family_rows,
                    log_probs=self._attack_arg1_log_probs(
                        family_row_states,
                        slot_context=self_stage_context.index_select(0, family_rows),
                        legal_mask=arg1_mask,
                    ),
                    mask=arg1_mask,
                )
        return plan, family_log_probs, arg0_log_probs, arg1_log_probs

    def evaluate_factorized_packed(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        actions: Tensor | None = None,
        same_family_reference_actions: Tensor | None = None,
        same_family_reference_families: Tensor | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
    ) -> _FactorizedEvaluationResult:
        row_states, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )
        plan, family_log_probs, arg0_log_probs, arg1_log_probs = self._factorized_distributions(
            row_states,
            legal_actions=legal_actions,
            observation_context=resolved_context,
        )
        row_count = int(row_states.shape[0])
        entropy = _masked_entropy_from_log_probs(family_log_probs, plan.family_mask)
        play_slot_log_probs = None
        move_source_log_probs = None
        move_slot_log_probs = None
        attack_slot_log_probs = None
        attack_type_log_probs = None
        for family_id, arg0_entry in arg0_log_probs.items():
            family_rows = arg0_entry.row_indices
            family_prob = torch.exp(family_log_probs.index_select(0, family_rows)[:, family_id])
            arg0_entropy = _masked_entropy_from_log_probs(arg0_entry.log_probs, arg0_entry.mask)
            entropy.index_add_(0, family_rows, family_prob * arg0_entropy)
            arg1_entry = arg1_log_probs.get(family_id)
            if arg1_entry is None or plan.family_plans[family_id].arg1_mask is None:
                if family_id == self._attack_family_id:
                    attack_slot_log_probs = _scatter_factorized_row_values(
                        row_count,
                        family_rows,
                        arg0_entry.log_probs,
                    )
                continue
            arg1_entropy = _masked_entropy_from_log_probs(
                arg1_entry.log_probs.reshape(-1, arg1_entry.log_probs.shape[-1]),
                arg1_entry.mask.reshape(-1, arg1_entry.mask.shape[-1]),
            ).reshape(arg1_entry.log_probs.shape[0], arg1_entry.log_probs.shape[1])
            arg0_probs = torch.where(
                arg0_entry.mask, torch.exp(arg0_entry.log_probs), torch.zeros_like(arg0_entry.log_probs)
            )
            entropy.index_add_(0, family_rows, family_prob * (arg0_probs * arg1_entropy).sum(dim=1))
            if family_id == self._play_character_family_id:
                play_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
            elif family_id == self._main_move_family_id:
                move_source_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    arg0_entry.log_probs,
                )
                move_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
            elif family_id == self._attack_family_id:
                attack_slot_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    arg0_entry.log_probs,
                )
                attack_type_log_probs = _scatter_factorized_row_values(
                    row_count,
                    family_rows,
                    torch.logsumexp(arg0_entry.log_probs.unsqueeze(-1) + arg1_entry.log_probs, dim=1),
                )
        action_logp = None
        if actions is not None:
            flat_actions = actions.reshape(-1).to(device=row_states.device, dtype=torch.long)
            selected_family = self._family_ids.index_select(0, flat_actions).to(dtype=torch.long)
            selected_arg0 = self._action_arg0.index_select(0, flat_actions).to(dtype=torch.long)
            selected_arg1 = self._action_arg1.index_select(0, flat_actions).to(dtype=torch.long)
            action_logp = family_log_probs.gather(1, selected_family.unsqueeze(1)).squeeze(1)
            for family_id, arg0_entry in arg0_log_probs.items():
                family_rows = selected_family == int(family_id)
                if not bool(family_rows.any().item()):
                    continue
                row_indices = torch.nonzero(family_rows, as_tuple=False).squeeze(1)
                local_row_indices = _factorized_local_row_indices(arg0_entry.row_indices, row_indices)
                arg0_indices = selected_arg0.index_select(0, row_indices)
                action_logp[row_indices] = action_logp[row_indices] + arg0_entry.log_probs.index_select(
                    0, local_row_indices
                ).gather(
                    1,
                    arg0_indices.unsqueeze(1),
                ).squeeze(1)
                arg1_entry = arg1_log_probs.get(family_id)
                if arg1_entry is None:
                    continue
                arg1_indices = selected_arg1.index_select(0, row_indices)
                action_logp[row_indices] = action_logp[row_indices] + arg1_entry.log_probs.index_select(
                    0, local_row_indices
                ).gather(
                    1,
                    arg0_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, arg1_entry.log_probs.shape[-1]),
                ).squeeze(1).gather(1, arg1_indices.unsqueeze(1)).squeeze(1)
        top_action_ids = self._factorized_top_action_ids(
            plan=plan,
            family_log_probs=family_log_probs,
            arg0_log_probs=arg0_log_probs,
            arg1_log_probs=arg1_log_probs,
        )
        same_family_action_logp = None
        same_family_top_action_ids = None
        if same_family_reference_actions is not None and same_family_reference_families is not None:
            same_family_action_logp, same_family_top_action_ids = self._factorized_same_family_action_stats(
                plan=plan,
                arg0_log_probs=arg0_log_probs,
                arg1_log_probs=arg1_log_probs,
                reference_actions=same_family_reference_actions,
                reference_families=same_family_reference_families,
                dtype=row_states.dtype,
            )
        return _FactorizedEvaluationResult(
            values=row_states.new_zeros((row_count,)),
            action_logp=action_logp,
            entropy=entropy,
            family_log_probs=family_log_probs,
            play_slot_log_probs=play_slot_log_probs,
            move_source_log_probs=move_source_log_probs,
            move_slot_log_probs=move_slot_log_probs,
            attack_slot_log_probs=attack_slot_log_probs,
            attack_type_log_probs=attack_type_log_probs,
            top_action_ids=top_action_ids,
            same_family_action_logp=same_family_action_logp,
            same_family_top_action_ids=same_family_top_action_ids,
        )

    def _factorized_top_action_ids(
        self,
        *,
        plan: _FactorizedLegalityPlan,
        family_log_probs: Tensor,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
    ) -> Tensor:
        row_count = int(plan.row_count)
        family_count = int(family_log_probs.shape[-1])
        best_family_action_ids = torch.full(
            (row_count, family_count),
            -1,
            device=family_log_probs.device,
            dtype=torch.long,
        )
        best_family_conditional_logp = torch.full_like(family_log_probs, -torch.inf)
        for family_id, family_plan in plan.family_plans.items():
            family_rows = family_plan.row_indices.to(dtype=torch.long)
            if family_rows.numel() == 0:
                continue
            family_kind = int(self._family_arg_kind[int(family_id)].item())
            if family_kind == 0:
                best_family_action_ids[family_rows, family_id] = int(
                    self._family_noarg_action_ids[int(family_id)].item()
                )
                best_family_conditional_logp[family_rows, family_id] = 0.0
                continue
            arg0_entry = arg0_log_probs.get(int(family_id))
            if arg0_entry is None:
                continue
            row_arg0_log_probs = arg0_entry.log_probs
            if family_kind in {1, 5, 6}:
                best_arg0_logp, best_arg0 = row_arg0_log_probs.max(dim=1)
                resolved_ids = self._one_arg_action_ids[int(family_id)].to(
                    device=family_log_probs.device, dtype=torch.long
                )
                best_family_action_ids[family_rows, family_id] = resolved_ids.index_select(0, best_arg0)
                best_family_conditional_logp[family_rows, family_id] = best_arg0_logp
                continue
            arg1_entry = arg1_log_probs.get(int(family_id))
            if arg1_entry is None:
                continue
            joint_log_probs = row_arg0_log_probs.unsqueeze(-1) + arg1_entry.log_probs
            flat_joint = joint_log_probs.reshape(joint_log_probs.shape[0], -1)
            best_joint_logp, best_joint = flat_joint.max(dim=1)
            arg1_size = int(joint_log_probs.shape[-1])
            best_arg0 = best_joint // arg1_size
            best_arg1 = best_joint % arg1_size
            resolved_ids = self._two_arg_action_ids[int(family_id)].to(device=family_log_probs.device, dtype=torch.long)
            best_family_action_ids[family_rows, family_id] = resolved_ids[best_arg0, best_arg1]
            best_family_conditional_logp[family_rows, family_id] = best_joint_logp
        total_logp = torch.where(
            best_family_action_ids >= 0,
            family_log_probs + best_family_conditional_logp,
            torch.full_like(family_log_probs, -torch.inf),
        )
        best_family = total_logp.argmax(dim=1)
        return best_family_action_ids.gather(1, best_family.unsqueeze(1)).squeeze(1)

    def _factorized_same_family_action_stats(
        self,
        *,
        plan: _FactorizedLegalityPlan,
        arg0_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        arg1_log_probs: Mapping[int, _FactorizedConditionalLogProbs],
        reference_actions: Tensor,
        reference_families: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        action_ids = reference_actions.reshape(-1).to(device=self._family_ids.device, dtype=torch.long)
        family_ids = reference_families.reshape(-1).to(device=self._family_ids.device, dtype=torch.long)
        row_count = int(plan.row_count)
        same_family_action_logp = torch.full(
            (row_count,),
            -torch.inf,
            device=self._family_ids.device,
            dtype=dtype,
        )
        same_family_top_action_ids = torch.full(
            (row_count,),
            -1,
            device=self._family_ids.device,
            dtype=torch.long,
        )
        if action_ids.numel() != row_count or family_ids.numel() != row_count or row_count == 0:
            return same_family_action_logp, same_family_top_action_ids
        valid_rows = (
            (action_ids >= 0)
            & (action_ids < self.action_dim)
            & (family_ids >= 0)
            & (family_ids < plan.family_mask.shape[1])
        )
        if not bool(valid_rows.any().item()):
            return same_family_action_logp, same_family_top_action_ids
        clamped_families = torch.clamp(family_ids, min=0, max=max(int(plan.family_mask.shape[1]) - 1, 0))
        valid_rows = valid_rows & plan.family_mask.gather(1, clamped_families.unsqueeze(1)).squeeze(1)
        if not bool(valid_rows.any().item()):
            return same_family_action_logp, same_family_top_action_ids
        valid_row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(1)
        valid_action_ids = action_ids.index_select(0, valid_row_indices)
        valid_family_ids = family_ids.index_select(0, valid_row_indices)
        valid_action_family_ids = self._family_ids.index_select(0, valid_action_ids)
        valid_action_arg0 = self._action_arg0.index_select(0, valid_action_ids)
        valid_action_arg1 = self._action_arg1.index_select(0, valid_action_ids)
        for family_id in torch.unique(valid_family_ids, sorted=True).tolist():
            family_rows = valid_family_ids == int(family_id)
            if not bool(family_rows.any().item()):
                continue
            row_indices = valid_row_indices[family_rows]
            row_action_ids = valid_action_ids[family_rows]
            row_action_family_ids = valid_action_family_ids[family_rows]
            row_action_arg0 = valid_action_arg0[family_rows]
            row_action_arg1 = valid_action_arg1[family_rows]
            family_kind = int(self._family_arg_kind[int(family_id)].item())
            if family_kind == 0:
                resolved_id = int(self._family_noarg_action_ids[int(family_id)].item())
                same_family_top_action_ids[row_indices] = resolved_id
                supported = row_action_ids == resolved_id
                if bool(supported.any().item()):
                    same_family_action_logp[row_indices[supported]] = 0.0
                continue
            arg0_entry = arg0_log_probs.get(int(family_id))
            if arg0_entry is None:
                continue
            local_row_indices = _factorized_local_row_indices(arg0_entry.row_indices, row_indices)
            row_arg0_log_probs = arg0_entry.log_probs.index_select(0, local_row_indices)
            row_arg0_mask = arg0_entry.mask.index_select(0, local_row_indices)
            if family_kind in {1, 5, 6}:
                top_arg0 = row_arg0_log_probs.argmax(dim=1)
                resolved_ids = self._one_arg_action_ids[int(family_id)].to(device=row_indices.device, dtype=torch.long)
                same_family_top_action_ids[row_indices] = resolved_ids.index_select(0, top_arg0)
                supported = (row_action_family_ids == int(family_id)) & (row_action_arg0 >= 0)
                if bool(supported.any().item()):
                    gather_arg0 = torch.clamp(row_action_arg0, min=0)
                    supported = supported & row_arg0_mask.gather(1, gather_arg0.unsqueeze(1)).squeeze(1)
                if bool(supported.any().item()):
                    supported_arg0 = row_action_arg0[supported]
                    same_family_action_logp[row_indices[supported]] = (
                        row_arg0_log_probs[supported]
                        .gather(
                            1,
                            supported_arg0.unsqueeze(1),
                        )
                        .squeeze(1)
                    )
                continue
            arg1_entry = arg1_log_probs.get(int(family_id))
            if arg1_entry is None:
                continue
            row_arg1_log_probs = arg1_entry.log_probs.index_select(0, local_row_indices)
            row_arg1_mask = arg1_entry.mask.index_select(0, local_row_indices)
            joint_log_probs = row_arg0_log_probs.unsqueeze(-1) + row_arg1_log_probs
            flat_joint = joint_log_probs.reshape(joint_log_probs.shape[0], -1)
            top_joint = flat_joint.argmax(dim=1)
            arg1_size = int(joint_log_probs.shape[-1])
            top_arg0 = top_joint // arg1_size
            top_arg1 = top_joint % arg1_size
            resolved_ids = self._two_arg_action_ids[int(family_id)].to(device=row_indices.device, dtype=torch.long)
            same_family_top_action_ids[row_indices] = resolved_ids[top_arg0, top_arg1]
            supported = (row_action_family_ids == int(family_id)) & (row_action_arg0 >= 0) & (row_action_arg1 >= 0)
            if bool(supported.any().item()):
                gather_arg0 = torch.clamp(row_action_arg0, min=0)
                gather_arg1 = torch.clamp(row_action_arg1, min=0)
                supported = (
                    supported
                    & row_arg1_mask[
                        torch.arange(row_indices.shape[0], device=row_indices.device, dtype=torch.long),
                        gather_arg0,
                        gather_arg1,
                    ]
                )
            if bool(supported.any().item()):
                supported_arg0 = row_action_arg0[supported]
                supported_arg1 = row_action_arg1[supported]
                supported_rows = torch.arange(
                    row_indices.shape[0],
                    device=row_indices.device,
                    dtype=torch.long,
                )[supported]
                same_family_action_logp[row_indices[supported]] = (
                    row_arg0_log_probs[supported].gather(1, supported_arg0.unsqueeze(1)).squeeze(1)
                    + row_arg1_log_probs[supported_rows, supported_arg0, supported_arg1]
                )
        return same_family_action_logp, same_family_top_action_ids

    def sample_factorized_packed(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
        observation_context: Mapping[str, Tensor] | None = None,
        state_repr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        row_states, resolved_context = (
            (state_repr, dict(observation_context))
            if state_repr is not None and observation_context is not None
            else self._build_state_representation(latent, obs=obs, observation_context=observation_context)
        )
        plan, family_log_probs, arg0_log_probs, arg1_log_probs = self._factorized_distributions(
            row_states,
            legal_actions=legal_actions,
            observation_context=resolved_context,
        )
        family_actions, behavior_logp = _sample_masked_log_probs(
            family_log_probs,
            plan.family_mask,
            sample_seeds=sample_seeds.to(device=row_states.device, dtype=torch.long),
            default_index=max(self._pass_family_id, 0),
        )
        actions = torch.full((row_states.shape[0],), int(pass_action_id), device=row_states.device, dtype=torch.long)
        for family_id in range(int(self._family_arg_kind.shape[0])):
            family_rows = torch.nonzero(family_actions == int(family_id), as_tuple=False).squeeze(1)
            if family_rows.numel() == 0:
                continue
            kind = int(self._family_arg_kind[family_id].item())
            if kind == 0:
                resolved_ids = self._family_noarg_action_ids[family_id]
                actions[family_rows] = torch.where(
                    resolved_ids >= 0,
                    resolved_ids.to(device=row_states.device, dtype=torch.long).expand_as(family_rows),
                    torch.full_like(family_rows, int(pass_action_id), dtype=torch.long),
                )
                continue
            arg0_log_probs_family = arg0_log_probs.get(family_id)
            if arg0_log_probs_family is None:
                continue
            local_row_indices = _factorized_local_row_indices(arg0_log_probs_family.row_indices, family_rows)
            arg0_actions, arg0_logp = _sample_masked_log_probs(
                arg0_log_probs_family.log_probs.index_select(0, local_row_indices),
                arg0_log_probs_family.mask.index_select(0, local_row_indices),
                sample_seeds=_derived_sample_seeds(sample_seeds.index_select(0, family_rows), salt=0x9E3779B1),
                default_index=0,
            )
            behavior_logp[family_rows] = behavior_logp[family_rows] + arg0_logp
            if kind in {1, 5, 6}:
                resolved_ids = self._one_arg_action_ids[family_id].to(device=row_states.device, dtype=torch.long)
                action_ids = resolved_ids.index_select(0, arg0_actions)
                actions[family_rows] = torch.where(
                    action_ids >= 0,
                    action_ids,
                    torch.full_like(action_ids, int(pass_action_id)),
                )
                continue
            arg1_log_probs_family = arg1_log_probs.get(family_id)
            if arg1_log_probs_family is None:
                continue
            row_arg1_log_probs = arg1_log_probs_family.log_probs.index_select(0, local_row_indices)[
                torch.arange(family_rows.shape[0], device=row_states.device, dtype=torch.long),
                arg0_actions,
            ]
            row_arg1_mask = arg1_log_probs_family.mask.index_select(0, local_row_indices)[
                torch.arange(family_rows.shape[0], device=row_states.device, dtype=torch.long),
                arg0_actions,
            ]
            arg1_actions, arg1_logp = _sample_masked_log_probs(
                row_arg1_log_probs,
                row_arg1_mask,
                sample_seeds=_derived_sample_seeds(sample_seeds.index_select(0, family_rows), salt=0x85EBCA77),
                default_index=0,
            )
            behavior_logp[family_rows] = behavior_logp[family_rows] + arg1_logp
            resolved_ids = self._two_arg_action_ids[family_id].to(device=row_states.device, dtype=torch.long)
            action_ids = resolved_ids[arg0_actions, arg1_actions]
            actions[family_rows] = torch.where(
                action_ids >= 0,
                action_ids,
                torch.full_like(action_ids, int(pass_action_id)),
            )
        return actions, behavior_logp

    def _build_packed_scoring_plan(
        self,
        *,
        candidate_ids: Tensor,
        offsets: Tensor,
        candidate_meta: Tensor | None,
    ) -> _PackedScoringPlan:
        if candidate_meta is None:
            family_ids = self._family_ids.index_select(0, candidate_ids)
            arg0 = self._action_arg0.index_select(0, candidate_ids)
            arg1 = self._action_arg1.index_select(0, candidate_ids)
        else:
            family_ids = candidate_meta[:, 0].to(dtype=torch.long)
            arg0 = candidate_meta[:, 1].to(dtype=torch.long)
            arg1 = candidate_meta[:, 2].to(dtype=torch.long)
            meta_unused = torch.full_like(arg0, self._meta_unused)
            arg0 = torch.where(arg0 == meta_unused, torch.full_like(arg0, -1), arg0)
            arg1 = torch.where(arg1 == meta_unused, torch.full_like(arg1, -1), arg1)
        return _PackedScoringPlan(
            row_indices=_packed_row_indices(offsets),
            family_ids=family_ids,
            arg0=arg0,
            arg1=arg1,
        )

    def _partition_candidate_family_indices(
        self,
        family_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = family_ids.device
        play_mask = family_ids == self._play_character_family_id
        hand_mask = torch.zeros_like(play_mask)
        for family_id in self._hand_family_ids:
            hand_mask |= family_ids == family_id
        move_mask = family_ids == self._main_move_family_id
        attack_mask = family_ids == self._attack_family_id
        slot_mask = torch.zeros_like(play_mask)
        for family_id in self._slot_family_ids:
            slot_mask |= family_ids == family_id
        index_mask = torch.zeros_like(play_mask)
        for family_id in self._index_family_ids:
            index_mask |= family_ids == family_id
        default_mask = ~(play_mask | hand_mask | move_mask | attack_mask | slot_mask | index_mask)

        def _indices(mask: Tensor) -> Tensor:
            if not torch.any(mask):
                return torch.zeros((0,), device=device, dtype=torch.long)
            return torch.nonzero(mask, as_tuple=False).squeeze(1)

        return (
            _indices(play_mask),
            _indices(hand_mask),
            _indices(move_mask),
            _indices(attack_mask),
            _indices(slot_mask),
            _indices(index_mask),
            _indices(default_mask),
        )

    def _project_generic_index_features(
        self,
        index_values: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        valid = index_values >= 0
        embedded = _optional_embedding(self.generic_index_embedding, index_values).to(dtype=dtype)
        projected = self.generic_candidate_projection(embedded)
        return projected * valid.unsqueeze(1).to(dtype=dtype)

    def _score_candidates_chunked(
        self,
        state_repr: Tensor,
        row_indices: Tensor,
        candidate_ids: Tensor,
        observation_context: Mapping[str, Tensor],
        *,
        candidate_meta: Tensor | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if candidate_ids.numel() == 0:
            return state_repr.new_zeros((0,))
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "learner" and state_repr.device.type == "cuda":
            chunk_size = max(chunk_size, int(self._cuda_learner_candidate_scoring_chunk_size))
        for start in range(0, int(candidate_ids.numel()), chunk_size):
            end = min(start + chunk_size, int(candidate_ids.numel()))
            scores_chunks.append(
                self._score_candidates(
                    state_repr,
                    row_indices[start:end],
                    candidate_ids[start:end],
                    observation_context,
                    candidate_meta=None if candidate_meta is None else candidate_meta[start:end],
                    scoring_mode=resolved_mode,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _score_packed_candidates_chunked(
        self,
        state_repr: Tensor,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if scoring_plan.candidate_count == 0:
            return state_repr.new_zeros((0,))
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "learner" and state_repr.device.type == "cuda":
            chunk_size = max(chunk_size, int(self._cuda_learner_candidate_scoring_chunk_size))
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            scores_chunks.append(
                self._score_packed_candidates_plan(
                    state_repr,
                    scoring_plan.slice(start, end),
                    observation_context,
                    scoring_mode=resolved_mode,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _score_packed_candidates_plan(
        self,
        state_repr: Tensor,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        row_indices_long = scoring_plan.row_indices.to(dtype=torch.long)
        row_states = state_repr.index_select(0, row_indices_long)
        family_embeddings = self.family_embedding(scoring_plan.family_ids).to(dtype=row_states.dtype)
        scores = row_states.new_empty((scoring_plan.candidate_count,), dtype=row_states.dtype)
        public_bias_scale = self._public_heuristic_logit_bias_scale_for(scoring_mode)
        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]
        (
            play_indices,
            hand_indices,
            move_indices,
            attack_indices,
            slot_family_indices,
            index_family_indices,
            default_indices,
        ) = self._partition_candidate_family_indices(scoring_plan.family_ids)

        if play_indices.numel() > 0:
            play_rows = row_indices_long.index_select(0, play_indices)
            play_row_states = row_states.index_select(0, play_indices)
            play_hand_indices = scoring_plan.arg0.index_select(0, play_indices)
            play_stage_slots = scoring_plan.arg1.index_select(0, play_indices)
            play_hand_present, play_hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                play_rows,
                play_hand_indices,
                dtype=row_states.dtype,
            )
            play_target_context, play_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                play_rows,
                play_stage_slots,
            )
            play_scores = self._score_candidate_group(
                play_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, play_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        play_hand_card_embeddings,
                        (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, play_stage_slots).to(dtype=row_states.dtype),
                        (self._stage_slot_feature_offset, self._from_slot_feature_offset),
                    ),
                    (
                        play_target_context.to(dtype=row_states.dtype),
                        (self._play_target_context_offset, self._move_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (play_hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),
                    ((1.0 - play_target_numeric[:, :1]).to(dtype=row_states.dtype), (8,)),
                ),
                constant_numeric_ones=(1, 9),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                play_scores = self._apply_public_heuristic_bias(
                    play_scores,
                    self._play_public_heuristic_raw(
                        play_stage_slots,
                        play_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=scoring_plan.family_ids.index_select(0, play_indices),
                )
            scores.index_copy_(
                0,
                play_indices,
                play_scores,
            )

        if hand_indices.numel() > 0:
            hand_rows = row_indices_long.index_select(0, hand_indices)
            hand_row_states = row_states.index_select(0, hand_indices)
            hand_family_indices = scoring_plan.arg0.index_select(0, hand_indices)
            hand_present, hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                hand_rows,
                hand_family_indices,
                dtype=row_states.dtype,
            )
            hand_scores = self._score_candidate_group(
                hand_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, hand_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        hand_card_embeddings,
                        (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                    ),
                ),
                numeric_sections=((hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),),
                constant_numeric_ones=(8, 9),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                attackers_available, front_defenders = self._public_attack_profile(
                    self_stage_numeric,
                    opponent_stage_numeric,
                    dtype=row_states.dtype,
                )
                hand_scores = self._apply_public_heuristic_bias(
                    hand_scores,
                    self._hand_public_heuristic_raw(
                        scoring_plan.family_ids.index_select(0, hand_indices),
                        hand_family_indices,
                        attackers_available=attackers_available.index_select(0, hand_rows),
                        front_defenders=front_defenders.index_select(0, hand_rows),
                        self_level_count=observation_context["self_level_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        self_clock_count=observation_context["self_clock_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=scoring_plan.family_ids.index_select(0, hand_indices),
                )
            scores.index_copy_(0, hand_indices, hand_scores)

        if move_indices.numel() > 0:
            move_rows = row_indices_long.index_select(0, move_indices)
            move_row_states = row_states.index_select(0, move_indices)
            move_from_slots = scoring_plan.arg0.index_select(0, move_indices)
            move_to_slots = scoring_plan.arg1.index_select(0, move_indices)
            move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_from_slots,
            )
            move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_to_slots,
            )
            move_scores = self._score_candidate_group(
                move_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, move_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, move_from_slots).to(dtype=row_states.dtype),
                        (self._from_slot_feature_offset, self._to_slot_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, move_to_slots).to(dtype=row_states.dtype),
                        (self._to_slot_feature_offset, self._attack_slot_feature_offset),
                    ),
                    (
                        move_source_context.to(dtype=row_states.dtype),
                        (self._move_source_context_offset, self._move_target_context_offset),
                    ),
                    (
                        move_target_context.to(dtype=row_states.dtype),
                        (self._move_target_context_offset, self._attack_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (move_source_numeric[:, :1].to(dtype=row_states.dtype), (7,)),
                    ((1.0 - move_target_numeric[:, :1]).to(dtype=row_states.dtype), (9,)),
                ),
                constant_numeric_ones=(2, 3, 8),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                move_scores = self._apply_public_heuristic_bias(
                    move_scores,
                    self._move_public_heuristic_raw(
                        move_from_slots,
                        move_to_slots,
                        move_source_numeric,
                        move_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=scoring_plan.family_ids.index_select(0, move_indices),
                )
            scores.index_copy_(
                0,
                move_indices,
                move_scores,
            )

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_row_states = row_states.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
            attack_source_context, attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                opponent_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            attack_scores = self._score_candidate_group(
                attack_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, attack_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, attack_slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        _optional_embedding(self.attack_type_embedding, attack_type_values).to(dtype=row_states.dtype),
                        (self._attack_type_feature_offset, self._play_target_context_offset),
                    ),
                    (
                        attack_source_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                    (
                        defender_context.to(dtype=row_states.dtype),
                        (self._defender_context_offset, self._numeric_feature_offset),
                    ),
                ),
                numeric_sections=((defender_numeric[:, :1].to(dtype=row_states.dtype), (10,)),),
                constant_numeric_ones=(4, 5, 8, 9),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                attack_scores = self._apply_public_heuristic_bias(
                    attack_scores,
                    self._attack_public_heuristic_raw(
                        attack_slot_values,
                        attack_type_values,
                        attack_source_numeric,
                        defender_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=scoring_plan.family_ids.index_select(0, attack_indices),
                )
            scores.index_copy_(
                0,
                attack_indices,
                attack_scores,
            )

        if slot_family_indices.numel() > 0:
            slot_rows = row_indices_long.index_select(0, slot_family_indices)
            slot_row_states = row_states.index_select(0, slot_family_indices)
            slot_family_ids = scoring_plan.family_ids.index_select(0, slot_family_indices)
            slot_values = scoring_plan.arg0.index_select(0, slot_family_indices)
            slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            slot_scores = self._score_candidate_group(
                slot_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, slot_family_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        slot_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                ),
                numeric_sections=((slot_numeric[:, :1].to(dtype=row_states.dtype), (7,)),),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                slot_scores = self._apply_public_heuristic_bias(
                    slot_scores,
                    self._slot_family_public_heuristic_raw(
                        slot_family_ids,
                        slot_values,
                        slot_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=slot_family_ids,
                )
            scores.index_copy_(0, slot_family_indices, slot_scores)

        if index_family_indices.numel() > 0:
            index_rows = row_indices_long.index_select(0, index_family_indices)
            index_row_states = row_states.index_select(0, index_family_indices)
            index_values = scoring_plan.arg0.index_select(0, index_family_indices)
            index_scores = self._score_candidate_group(
                index_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, index_family_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                    (
                        self._project_generic_index_features(index_values, dtype=row_states.dtype),
                        (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                    ),
                ),
                numeric_sections=((torch.clamp(index_values.to(dtype=row_states.dtype), min=0.0).unsqueeze(1), (6,)),),
                scoring_mode=scoring_mode,
            )
            if public_bias_scale > 0.0:
                index_scores = self._apply_public_heuristic_bias(
                    index_scores,
                    self._index_public_heuristic_raw(
                        scoring_plan.family_ids.index_select(0, index_family_indices),
                        index_values,
                        choice_page_start=observation_context["choice_page_start"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        choice_total=observation_context["choice_total"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=scoring_plan.family_ids.index_select(0, index_family_indices),
                )
            scores.index_copy_(0, index_family_indices, index_scores)

        if default_indices.numel() > 0:
            default_row_states = row_states.index_select(0, default_indices)
            default_generic_indices = scoring_plan.arg0.index_select(0, default_indices)
            default_scores = self._score_candidate_group(
                default_row_states,
                feature_sections=(
                    (
                        family_embeddings.index_select(0, default_indices),
                        (self._family_feature_offset, self._hand_card_feature_offset),
                    ),
                ),
                numeric_sections=(((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),),
                constant_numeric_ones=(8, 9),
                scoring_mode=scoring_mode,
            )
            default_family_ids = scoring_plan.family_ids.index_select(0, default_indices)
            if public_bias_scale > 0.0:
                default_scores = self._apply_public_heuristic_bias(
                    default_scores,
                    self._default_public_heuristic_raw(
                        default_family_ids,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=default_family_ids,
                )
            scores.index_copy_(
                0,
                default_indices,
                default_scores,
            )

        return scores + self.family_bias.index_select(0, scoring_plan.family_ids).to(dtype=row_states.dtype)

    def _project_candidate_sections(
        self,
        *,
        feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
        numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
        constant_numeric_ones: Sequence[int] = (),
        scoring_mode: str = "auto",
    ) -> Tensor:
        if not isinstance(self.candidate_projection[0], nn.Linear):
            raise RuntimeError("structured candidate projection must begin with nn.Linear")
        linear = self.candidate_projection[0]
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "actor":
            inputs: list[Tensor] = []
            weight_blocks: list[Tensor] = []
            for tensor, (start, end) in feature_sections:
                if tensor.numel() == 0:
                    continue
                inputs.append(tensor)
                weight_blocks.append(linear.weight[:, start:end])
            for tensor, numeric_indices in numeric_sections:
                if tensor.numel() == 0:
                    continue
                inputs.append(tensor)
                column_indices = torch.as_tensor(
                    [self._numeric_feature_offset + int(index) for index in numeric_indices],
                    device=linear.weight.device,
                    dtype=torch.long,
                )
                weight_blocks.append(linear.weight.index_select(1, column_indices))
            if not inputs or not weight_blocks:
                raise ValueError("structured candidate projection requires at least one feature section")
            projected = F.linear(
                torch.cat(inputs, dim=1),
                torch.cat(weight_blocks, dim=1),
                linear.bias,
            )
            if constant_numeric_ones:
                constant_columns = torch.as_tensor(
                    [self._numeric_feature_offset + int(index) for index in constant_numeric_ones],
                    device=linear.weight.device,
                    dtype=torch.long,
                )
                projected = projected + linear.weight.index_select(1, constant_columns).sum(dim=1).to(
                    dtype=projected.dtype
                )
            for module in self.candidate_projection[1:]:
                projected = module(projected)
            return projected
        projected: Tensor | None = None
        for tensor, (start, end) in feature_sections:
            if tensor.numel() == 0:
                continue
            if projected is None:
                projected = tensor.new_zeros((tensor.shape[0], linear.out_features))
                if linear.bias is not None:
                    projected = projected + linear.bias.to(dtype=projected.dtype)
            projected = projected + F.linear(tensor, linear.weight[:, start:end], None)
        for tensor, numeric_indices in numeric_sections:
            if tensor.numel() == 0:
                continue
            if projected is None:
                projected = tensor.new_zeros((tensor.shape[0], linear.out_features))
                if linear.bias is not None:
                    projected = projected + linear.bias.to(dtype=projected.dtype)
            column_indices = torch.as_tensor(
                [self._numeric_feature_offset + int(index) for index in numeric_indices],
                device=linear.weight.device,
                dtype=torch.long,
            )
            projected = projected + F.linear(tensor, linear.weight.index_select(1, column_indices), None)
        if projected is None:
            raise ValueError("structured candidate projection requires at least one feature section")
        if constant_numeric_ones:
            constant_columns = torch.as_tensor(
                [self._numeric_feature_offset + int(index) for index in constant_numeric_ones],
                device=linear.weight.device,
                dtype=torch.long,
            )
            projected = projected + linear.weight.index_select(1, constant_columns).sum(dim=1).to(dtype=projected.dtype)
        for module in self.candidate_projection[1:]:
            projected = module(projected)
        return projected

    def _score_candidate_group(
        self,
        row_states: Tensor,
        *,
        feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
        numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
        constant_numeric_ones: Sequence[int] = (),
        scoring_mode: str = "auto",
    ) -> Tensor:
        if row_states.numel() == 0:
            return row_states.new_zeros((0,))
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        candidate_repr = self._project_candidate_sections(
            feature_sections=feature_sections,
            numeric_sections=numeric_sections,
            constant_numeric_ones=constant_numeric_ones,
            scoring_mode=resolved_mode,
        )
        if resolved_mode == "actor":
            return (
                self.joint_scorer(torch.cat([row_states, candidate_repr], dim=1)).squeeze(-1).to(dtype=row_states.dtype)
            )
        if not isinstance(self.joint_scorer[0], nn.Linear):
            raise RuntimeError("structured joint scorer must begin with nn.Linear")
        joint_linear = self.joint_scorer[0]
        state_width = row_states.shape[1]
        joint_hidden = F.linear(row_states, joint_linear.weight[:, :state_width], joint_linear.bias)
        joint_hidden = joint_hidden + F.linear(candidate_repr, joint_linear.weight[:, state_width:], None)
        for module in self.joint_scorer[1:]:
            joint_hidden = module(joint_hidden)
        return joint_hidden.squeeze(-1).to(dtype=row_states.dtype)

    def _score_candidates(
        self,
        state_repr: Tensor,
        row_indices: Tensor,
        candidate_ids: Tensor,
        observation_context: Mapping[str, Tensor],
        candidate_meta: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> Tensor:
        row_indices_long = row_indices.to(dtype=torch.long)
        row_states = state_repr.index_select(0, row_indices_long)
        hand_indices: Tensor | None = None
        stage_slots: Tensor | None = None
        from_slots: Tensor | None = None
        to_slots: Tensor | None = None
        attack_slots: Tensor | None = None
        attack_types: Tensor | None = None
        generic_indices: Tensor | None = None
        meta_arg0: Tensor | None = None
        meta_arg1: Tensor | None = None
        if candidate_meta is None:
            (
                family_ids,
                hand_indices,
                stage_slots,
                from_slots,
                to_slots,
                attack_slots,
                attack_types,
                generic_indices,
            ) = self._resolve_candidate_components(candidate_ids, None)
        else:
            family_ids = candidate_meta[:, 0].to(dtype=torch.long)
            meta_arg0 = candidate_meta[:, 1].to(dtype=torch.long)
            meta_arg1 = candidate_meta[:, 2].to(dtype=torch.long)
            meta_arg0 = torch.where(meta_arg0 == self._meta_unused, torch.full_like(meta_arg0, -1), meta_arg0)
            meta_arg1 = torch.where(meta_arg1 == self._meta_unused, torch.full_like(meta_arg1, -1), meta_arg1)
        family_embeddings = self.family_embedding(family_ids).to(dtype=row_states.dtype)
        scores = row_states.new_empty((candidate_ids.shape[0],), dtype=row_states.dtype)
        public_bias_scale = self._public_heuristic_logit_bias_scale_for(scoring_mode)
        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]

        play_mask = family_ids == self._play_character_family_id
        if torch.any(play_mask):
            play_rows = row_indices_long[play_mask]
            play_row_states = row_states[play_mask]
            play_hand_indices = meta_arg0[play_mask] if meta_arg0 is not None else hand_indices[play_mask]
            play_stage_slots = meta_arg1[play_mask] if meta_arg1 is not None else stage_slots[play_mask]
            play_hand_present, play_hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                play_rows,
                play_hand_indices,
                dtype=row_states.dtype,
            )
            play_target_context, play_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                play_rows,
                play_stage_slots,
            )
            play_scores = self._score_candidate_group(
                play_row_states,
                feature_sections=(
                    (family_embeddings[play_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (play_hand_card_embeddings, (self._hand_card_feature_offset, self._stage_slot_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, play_stage_slots).to(dtype=row_states.dtype),
                        (self._stage_slot_feature_offset, self._from_slot_feature_offset),
                    ),
                    (
                        play_target_context.to(dtype=row_states.dtype),
                        (self._play_target_context_offset, self._move_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (play_hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),
                    ((1.0 - play_target_numeric[:, :1]).to(dtype=row_states.dtype), (8,)),
                ),
                constant_numeric_ones=(1, 9),
            )
            if public_bias_scale > 0.0:
                play_scores = self._apply_public_heuristic_bias(
                    play_scores,
                    self._play_public_heuristic_raw(
                        play_stage_slots,
                        play_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[play_mask],
                )
            scores[play_mask] = play_scores

        hand_family_ids = (
            self._main_event_family_id,
            self._clock_from_hand_family_id,
            self._climax_play_family_id,
            self._mulligan_select_family_id,
        )
        hand_mask = torch.zeros_like(play_mask)
        for family_id in hand_family_ids:
            if family_id >= 0:
                hand_mask |= family_ids == family_id
        if torch.any(hand_mask):
            hand_rows = row_indices_long[hand_mask]
            hand_row_states = row_states[hand_mask]
            hand_family_indices = meta_arg0[hand_mask] if meta_arg0 is not None else hand_indices[hand_mask]
            hand_present, hand_card_embeddings = self._gather_hand_embeddings_from_rows(
                observation_context["hand_ids"],
                hand_rows,
                hand_family_indices,
                dtype=row_states.dtype,
            )
            scores[hand_mask] = self._score_candidate_group(
                hand_row_states,
                feature_sections=(
                    (family_embeddings[hand_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (hand_card_embeddings, (self._hand_card_feature_offset, self._stage_slot_feature_offset)),
                ),
                numeric_sections=((hand_present.to(dtype=row_states.dtype).unsqueeze(1), (0,)),),
                constant_numeric_ones=(8, 9),
            )
            if public_bias_scale > 0.0:
                attackers_available, front_defenders = self._public_attack_profile(
                    self_stage_numeric,
                    opponent_stage_numeric,
                    dtype=row_states.dtype,
                )
                scores[hand_mask] = self._apply_public_heuristic_bias(
                    scores[hand_mask],
                    self._hand_public_heuristic_raw(
                        family_ids[hand_mask],
                        hand_family_indices,
                        attackers_available=attackers_available.index_select(0, hand_rows),
                        front_defenders=front_defenders.index_select(0, hand_rows),
                        self_level_count=observation_context["self_level_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        self_clock_count=observation_context["self_clock_count"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, hand_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[hand_mask],
                )

        move_mask = family_ids == self._main_move_family_id
        if torch.any(move_mask):
            move_rows = row_indices_long[move_mask]
            move_row_states = row_states[move_mask]
            move_from_slots = meta_arg0[move_mask] if meta_arg0 is not None else from_slots[move_mask]
            move_to_slots = meta_arg1[move_mask] if meta_arg1 is not None else to_slots[move_mask]
            move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_from_slots,
            )
            move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_to_slots,
            )
            move_scores = self._score_candidate_group(
                move_row_states,
                feature_sections=(
                    (family_embeddings[move_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, move_from_slots).to(dtype=row_states.dtype),
                        (self._from_slot_feature_offset, self._to_slot_feature_offset),
                    ),
                    (
                        _optional_embedding(self.slot_embedding, move_to_slots).to(dtype=row_states.dtype),
                        (self._to_slot_feature_offset, self._attack_slot_feature_offset),
                    ),
                    (
                        move_source_context.to(dtype=row_states.dtype),
                        (self._move_source_context_offset, self._move_target_context_offset),
                    ),
                    (
                        move_target_context.to(dtype=row_states.dtype),
                        (self._move_target_context_offset, self._attack_source_context_offset),
                    ),
                ),
                numeric_sections=(
                    (move_source_numeric[:, :1].to(dtype=row_states.dtype), (7,)),
                    ((1.0 - move_target_numeric[:, :1]).to(dtype=row_states.dtype), (9,)),
                ),
                constant_numeric_ones=(2, 3, 8),
            )
            if public_bias_scale > 0.0:
                move_scores = self._apply_public_heuristic_bias(
                    move_scores,
                    self._move_public_heuristic_raw(
                        move_from_slots,
                        move_to_slots,
                        move_source_numeric,
                        move_target_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[move_mask],
                )
            scores[move_mask] = move_scores

        attack_mask = family_ids == self._attack_family_id
        if torch.any(attack_mask):
            attack_rows = row_indices_long[attack_mask]
            attack_row_states = row_states[attack_mask]
            attack_slot_values = meta_arg0[attack_mask] if meta_arg0 is not None else attack_slots[attack_mask]
            attack_type_values = meta_arg1[attack_mask] if meta_arg1 is not None else attack_types[attack_mask]
            attack_source_context, attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                opponent_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            attack_scores = self._score_candidate_group(
                attack_row_states,
                feature_sections=(
                    (family_embeddings[attack_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, attack_slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        _optional_embedding(self.attack_type_embedding, attack_type_values).to(dtype=row_states.dtype),
                        (self._attack_type_feature_offset, self._play_target_context_offset),
                    ),
                    (
                        attack_source_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                    (
                        defender_context.to(dtype=row_states.dtype),
                        (self._defender_context_offset, self._numeric_feature_offset),
                    ),
                ),
                numeric_sections=((defender_numeric[:, :1].to(dtype=row_states.dtype), (10,)),),
                constant_numeric_ones=(4, 5, 8, 9),
            )
            if public_bias_scale > 0.0:
                attack_scores = self._apply_public_heuristic_bias(
                    attack_scores,
                    self._attack_public_heuristic_raw(
                        attack_slot_values,
                        attack_type_values,
                        attack_source_numeric,
                        defender_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[attack_mask],
                )
            scores[attack_mask] = attack_scores

        slot_mask = torch.zeros_like(play_mask)
        for family_id in self._slot_family_ids:
            slot_mask |= family_ids == family_id
        if torch.any(slot_mask):
            slot_rows = row_indices_long[slot_mask]
            slot_row_states = row_states[slot_mask]
            slot_values = meta_arg0[slot_mask] if meta_arg0 is not None else attack_slots[slot_mask]
            slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            slot_scores = self._score_candidate_group(
                slot_row_states,
                feature_sections=(
                    (family_embeddings[slot_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        _optional_embedding(self.slot_embedding, slot_values).to(dtype=row_states.dtype),
                        (self._attack_slot_feature_offset, self._attack_type_feature_offset),
                    ),
                    (
                        slot_context.to(dtype=row_states.dtype),
                        (self._attack_source_context_offset, self._defender_context_offset),
                    ),
                ),
                numeric_sections=((slot_numeric[:, :1].to(dtype=row_states.dtype), (7,)),),
            )
            if public_bias_scale > 0.0:
                slot_scores = self._apply_public_heuristic_bias(
                    slot_scores,
                    self._slot_family_public_heuristic_raw(
                        family_ids[slot_mask],
                        slot_values,
                        slot_numeric,
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[slot_mask],
                )
            scores[slot_mask] = slot_scores

        index_mask = torch.zeros_like(play_mask)
        for family_id in self._index_family_ids:
            index_mask |= family_ids == family_id
        if torch.any(index_mask):
            index_rows = row_indices_long[index_mask]
            index_row_states = row_states[index_mask]
            index_values = meta_arg0[index_mask] if meta_arg0 is not None else generic_indices[index_mask]
            scores[index_mask] = self._score_candidate_group(
                index_row_states,
                feature_sections=(
                    (family_embeddings[index_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                    (
                        self._project_generic_index_features(index_values, dtype=row_states.dtype),
                        (self._hand_card_feature_offset, self._stage_slot_feature_offset),
                    ),
                ),
                numeric_sections=((torch.clamp(index_values.to(dtype=row_states.dtype), min=0.0).unsqueeze(1), (6,)),),
            )
            if public_bias_scale > 0.0:
                scores[index_mask] = self._apply_public_heuristic_bias(
                    scores[index_mask],
                    self._index_public_heuristic_raw(
                        family_ids[index_mask],
                        index_values,
                        choice_page_start=observation_context["choice_page_start"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        choice_total=observation_context["choice_total"]
                        .to(device=row_states.device, dtype=row_states.dtype)
                        .index_select(0, index_rows),
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[index_mask],
                )

        default_mask = ~(play_mask | hand_mask | move_mask | attack_mask | slot_mask | index_mask)
        if torch.any(default_mask):
            default_row_states = row_states[default_mask]
            default_generic_indices = (
                meta_arg0[default_mask] if meta_arg0 is not None else generic_indices[default_mask]
            )
            default_scores = self._score_candidate_group(
                default_row_states,
                feature_sections=(
                    (family_embeddings[default_mask], (self._family_feature_offset, self._hand_card_feature_offset)),
                ),
                numeric_sections=(((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),),
                constant_numeric_ones=(8, 9),
            )
            if public_bias_scale > 0.0:
                default_scores = self._apply_public_heuristic_bias(
                    default_scores,
                    self._default_public_heuristic_raw(
                        family_ids[default_mask],
                        dtype=row_states.dtype,
                    ),
                    scale=public_bias_scale,
                    family_ids=family_ids[default_mask],
                )
            scores[default_mask] = default_scores

        return scores + self.family_bias.index_select(0, family_ids).to(dtype=row_states.dtype)
