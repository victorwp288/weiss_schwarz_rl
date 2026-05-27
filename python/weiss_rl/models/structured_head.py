"""Structured legal-action policy head for the policy/value model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.card_table import card_feature_table
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.core.observation_layout import ObservationSlice
from weiss_rl.eval.heuristic_public import heuristic_public_scoring_profile
from weiss_rl.models.action_tables import (
    build_factorized_action_lookup_tables,
    build_structured_action_component_tables,
)
from weiss_rl.models.candidate_components import (
    CandidateComponentFamilyIds,
    resolve_candidate_components,
)
from weiss_rl.models.candidate_projection import (
    project_candidate_sections,
    score_candidate_group,
)
from weiss_rl.models.dense_scoring import StructuredDenseScoringMixin
from weiss_rl.models.factorized_scoring import StructuredFactorizedScoringMixin
from weiss_rl.models.feature_gathering import (
    gather_stage_features,
    gather_stage_features_for_rows,
    slot_component,
)
from weiss_rl.models.layers import build_mlp_stack as _build_mlp_stack
from weiss_rl.models.observation_context import (
    encode_observation_context,
    encode_stage_slice,
    extract_card_vector,
    extract_header_scalar,
    extract_scalar_feature,
)
from weiss_rl.models.observation_contract import StructuredObservationContract
from weiss_rl.models.packed_scoring import StructuredPackedScoringMixin
from weiss_rl.models.public_heuristic_scoring import StructuredPublicHeuristicScoringMixin
from weiss_rl.models.public_heuristics import public_heuristic_slot_preference_array
from weiss_rl.models.tensor_ops import (
    bucket_card_ids,
    negative_logits_fill_value,
    optional_embedding,
)

_StructuredObservationContract = StructuredObservationContract
_bucket_card_ids = bucket_card_ids
_optional_embedding = optional_embedding
_negative_logits_fill_value = negative_logits_fill_value


class _StructuredLegalActionHead(
    StructuredDenseScoringMixin,
    StructuredPackedScoringMixin,
    StructuredFactorizedScoringMixin,
    StructuredPublicHeuristicScoringMixin,
    nn.Module,
):
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

        action_tables = build_structured_action_component_tables(
            action_catalog=action_catalog,
            action_dim=int(self.action_dim),
            family_index=family_index,
            attack_type_index=attack_type_index,
        )

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
        final_scorer = nn.Linear(state_width, 1)
        nn.init.zeros_(final_scorer.weight)
        nn.init.zeros_(final_scorer.bias)
        scorer_layers.append(final_scorer)
        self.joint_scorer = nn.Sequential(*scorer_layers)
        self.family_bias = nn.Parameter(torch.zeros(max(len(family_names), 1)))
        self._candidate_scoring_chunk_size = int(candidate_scoring_chunk_size)
        self._cuda_learner_candidate_scoring_chunk_size = int(cuda_learner_candidate_scoring_chunk_size)
        self.register_buffer("_family_ids", torch.as_tensor(action_tables.family_ids, dtype=torch.long))
        self.register_buffer("_action_arg0", torch.as_tensor(action_tables.action_arg0, dtype=torch.long))
        self.register_buffer("_action_arg1", torch.as_tensor(action_tables.action_arg1, dtype=torch.long))
        self.register_buffer("_hand_indices", torch.as_tensor(action_tables.hand_indices, dtype=torch.long))
        self.register_buffer("_stage_slots", torch.as_tensor(action_tables.stage_slots, dtype=torch.long))
        self.register_buffer("_from_slots", torch.as_tensor(action_tables.from_slots, dtype=torch.long))
        self.register_buffer("_to_slots", torch.as_tensor(action_tables.to_slots, dtype=torch.long))
        self.register_buffer("_attack_slots", torch.as_tensor(action_tables.attack_slots, dtype=torch.long))
        self.register_buffer("_attack_types", torch.as_tensor(action_tables.attack_types, dtype=torch.long))
        self.register_buffer("_generic_indices", torch.as_tensor(action_tables.generic_indices, dtype=torch.long))

        family_count = max(len(family_names), 1)
        factorized_tables = build_factorized_action_lookup_tables(
            action_dim=int(self.action_dim),
            family_count=family_count,
            family_index=family_index,
            component_tables=action_tables,
        )
        generic_embed_dim = max(8, min(24, action_feature_width // 5))
        self.generic_index_embedding = nn.Embedding(int(factorized_tables.max_arg0) + 1, generic_embed_dim)
        self.generic_candidate_projection = _build_mlp_stack(
            input_dim=generic_embed_dim,
            width=card_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self.family_head = nn.Linear(state_width, family_count)
        nn.init.zeros_(self.family_head.weight)
        nn.init.zeros_(self.family_head.bias)
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
        self.register_buffer("_family_arg_kind", torch.as_tensor(factorized_tables.family_arg_kind, dtype=torch.long))
        self.register_buffer(
            "_family_arg0_size",
            torch.as_tensor(factorized_tables.family_arg0_size, dtype=torch.long),
        )
        self.register_buffer(
            "_family_arg1_size",
            torch.as_tensor(factorized_tables.family_arg1_size, dtype=torch.long),
        )
        self.register_buffer(
            "_family_noarg_action_ids",
            torch.as_tensor(factorized_tables.family_noarg_action_ids, dtype=torch.long),
        )
        self.register_buffer(
            "_one_arg_action_ids",
            torch.as_tensor(factorized_tables.one_arg_action_ids, dtype=torch.long),
        )
        self.register_buffer(
            "_two_arg_action_ids",
            torch.as_tensor(factorized_tables.two_arg_action_ids, dtype=torch.long),
        )
        self._slot_family_ids = factorized_tables.slot_family_ids
        self._index_family_ids = factorized_tables.index_family_ids
        self.register_buffer(
            "_public_slot_preference",
            torch.as_tensor(public_heuristic_slot_preference_array(self._stage_slot_count), dtype=torch.float32),
            persistent=False,
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
        return encode_observation_context(
            obs_batch=obs_batch,
            observation_contract=self._observation_contract,
            slot_context_dim=int(self._slot_context_dim),
            stage_slot_count=int(self._stage_slot_count),
            card_representation=self._card_representation,
            hand_summary_projection=self.hand_summary_projection,
            slot_encoder=self.slot_encoder,
        )

    def _extract_scalar_feature(
        self,
        obs_batch: Tensor,
        slice_spec: ObservationSlice | None,
    ) -> Tensor:
        return extract_scalar_feature(obs_batch, slice_spec)

    def _extract_header_scalar(
        self,
        obs_batch: Tensor,
        index: int | None,
    ) -> Tensor:
        return extract_header_scalar(obs_batch, index)

    def _encode_stage_slice(
        self,
        obs_batch: Tensor,
        stage_slice: ObservationSlice | None,
    ) -> tuple[Tensor, Tensor]:
        return encode_stage_slice(
            obs_batch=obs_batch,
            stage_slice=stage_slice,
            observation_contract=self._observation_contract,
            stage_slot_count=int(self._stage_slot_count),
            slot_context_dim=int(self._slot_context_dim),
            card_representation=self._card_representation,
            slot_encoder=self.slot_encoder,
        )

    def _resolve_scoring_mode(self, scoring_mode: str) -> str:
        resolved_mode = str(scoring_mode).strip().lower()
        if resolved_mode == "auto":
            return "actor" if not torch.is_grad_enabled() else "learner"
        if resolved_mode not in {"actor", "learner"}:
            raise ValueError("scoring_mode must be one of: auto, actor, learner")
        return resolved_mode

    def _project_candidate_sections(
        self,
        *,
        feature_sections: Sequence[tuple[Tensor, tuple[int, int]]],
        numeric_sections: Sequence[tuple[Tensor, Sequence[int]]] = (),
        constant_numeric_ones: Sequence[int] = (),
        scoring_mode: str = "auto",
    ) -> Tensor:
        return project_candidate_sections(
            candidate_projection=self.candidate_projection,
            numeric_feature_offset=self._numeric_feature_offset,
            feature_sections=feature_sections,
            numeric_sections=numeric_sections,
            constant_numeric_ones=constant_numeric_ones,
            scoring_mode=self._resolve_scoring_mode(scoring_mode),
        )

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
        return score_candidate_group(
            row_states,
            candidate_projection=self.candidate_projection,
            joint_scorer=self.joint_scorer,
            numeric_feature_offset=self._numeric_feature_offset,
            feature_sections=feature_sections,
            numeric_sections=numeric_sections,
            constant_numeric_ones=constant_numeric_ones,
            scoring_mode=resolved_mode,
        )

    def _resolve_candidate_components(
        self,
        candidate_ids: Tensor,
        candidate_meta: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return resolve_candidate_components(
            candidate_ids,
            candidate_meta,
            family_ids_by_action=self._family_ids,
            hand_indices_by_action=self._hand_indices,
            stage_slots_by_action=self._stage_slots,
            from_slots_by_action=self._from_slots,
            to_slots_by_action=self._to_slots,
            attack_slots_by_action=self._attack_slots,
            attack_types_by_action=self._attack_types,
            generic_indices_by_action=self._generic_indices,
            meta_unused=int(self._meta_unused),
            family_ids=CandidateComponentFamilyIds(
                play_character=int(self._play_character_family_id),
                main_event=int(self._main_event_family_id),
                clock_from_hand=int(self._clock_from_hand_family_id),
                climax_play=int(self._climax_play_family_id),
                mulligan_select=int(self._mulligan_select_family_id),
                main_move=int(self._main_move_family_id),
                attack=int(self._attack_family_id),
                choice_select=int(self._choice_select_family_id),
                level_up=int(self._level_up_family_id),
                trigger_order=int(self._trigger_order_family_id),
            ),
        )

    def _gather_hand_embeddings_from_rows(
        self,
        hand_ids: Tensor,
        row_indices: Tensor,
        hand_indices: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if hand_ids.shape[1] == 0:
            return (
                torch.zeros_like(hand_indices, dtype=torch.bool),
                hand_ids.new_zeros((hand_indices.shape[0], self.card_embedding.embedding_dim), dtype=dtype),
            )
        hand_present = (hand_indices >= 0) & (hand_indices < hand_ids.shape[1])
        if not torch.any(hand_present):
            return (
                hand_present,
                hand_ids.new_zeros((hand_indices.shape[0], self.card_embedding.embedding_dim), dtype=dtype),
            )
        safe_rows = torch.where(hand_present, row_indices, torch.zeros_like(row_indices)).to(dtype=torch.long)
        safe_hand = torch.where(hand_present, hand_indices, torch.zeros_like(hand_indices)).to(dtype=torch.long)
        flat_indices = safe_rows * int(hand_ids.shape[1]) + safe_hand
        candidate_hand_ids = hand_ids.reshape(-1).index_select(0, flat_indices)
        hand_card_embeddings = self._card_representation(candidate_hand_ids, dtype=dtype)
        hand_position_embeddings = _optional_embedding(self.hand_position_embedding, hand_indices).to(dtype=dtype)
        hand_card_embeddings = hand_card_embeddings + hand_position_embeddings
        return hand_present, hand_card_embeddings * hand_present.unsqueeze(1).to(dtype=dtype)

    def _gather_stage_features_for_rows(
        self,
        slot_contexts: Tensor,
        slot_numeric: Tensor,
        row_indices: Tensor,
        slot_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gather_stage_features_for_rows(
            slot_contexts,
            slot_numeric,
            row_indices,
            slot_indices,
            stage_slot_count=int(self._stage_slot_count),
        )

    def _card_representation(self, card_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
        bucketed_ids = _bucket_card_ids(card_ids, vocab_size=self._card_vocab_size)
        learned = self.card_embedding(bucketed_ids).to(dtype=dtype)
        if self.card_feature_projection is None or self._card_static_features.numel() == 0:
            return learned
        flat_ids = bucketed_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        static_features = self._card_static_features.index_select(0, unique_ids)
        projected_unique = self.card_feature_projection(static_features.to(dtype=dtype))
        projected = projected_unique.index_select(0, inverse).reshape(
            *bucketed_ids.shape,
            projected_unique.shape[-1],
        )
        return learned + projected.to(dtype=dtype)

    def _gather_stage_features(
        self,
        slot_contexts: Tensor,
        slot_numeric: Tensor,
        slot_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return gather_stage_features(
            slot_contexts,
            slot_numeric,
            slot_indices,
            stage_slot_count=int(self._stage_slot_count),
        )

    def _extract_card_vector(self, obs_batch: Tensor, observation_slice: ObservationSlice | None) -> Tensor:
        return extract_card_vector(obs_batch, observation_slice)

    def _slot_component(self, stage_values: Tensor, offset: int) -> Tensor:
        return slot_component(stage_values, int(offset))
