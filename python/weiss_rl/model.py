"""Torch recurrent actor-critic model."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.card_table import cached_runtime_card_table
from weiss_rl.models import typed_encoder as model_typed_encoder
from weiss_rl.models.action_plans import (
    FactorizedConditionalLogProbs,
    FactorizedEvaluationResult,
    FactorizedFamilyPlan,
    FactorizedLegalityPlan,
    PackedScoringPlan,
)
from weiss_rl.models.base import (
    SEAT_COUNT,
    STRUCTURED_V2_ENCODER_KIND,
    PolicyValueModelBaseMixin,
)
from weiss_rl.models.layers import build_mlp_stack as _build_mlp_stack
from weiss_rl.models.observation_contract import (
    CARD_ID_VECTOR_SLICE_NAMES,
    build_structured_observation_contract,
    header_field_index,
    slice_by_name,
)
from weiss_rl.models.policy_value_facade import StructuredLegalPolicyValueFacadeMixin
from weiss_rl.models.public_heuristics import (
    PUBLIC_HEURISTIC_BACK_ROW_SLOTS,
    PUBLIC_HEURISTIC_CENTER_SLOT,
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
)
from weiss_rl.models.sampling import (
    sample_masked_log_probs,
    sample_packed_action_scores,
)
from weiss_rl.models.structured_head import _StructuredLegalActionHead
from weiss_rl.models.tensor_ops import (
    bucket_card_ids,
    derived_sample_seeds,
    factorized_local_row_indices,
    masked_entropy_from_log_probs,
    masked_log_softmax,
    masked_max_pool,
    masked_mean_pool,
    negative_logits_fill_value,
    optional_embedding,
    packed_local_cdf,
    packed_row_indices,
    packed_row_log_z,
    scatter_factorized_row_values,
    uniform_from_seeds,
)

_CARD_ID_VECTOR_SLICE_NAMES = CARD_ID_VECTOR_SLICE_NAMES
_TypedObservationEncoder = model_typed_encoder.TypedObservationEncoder
_TypedPlayerBlockEncoder = model_typed_encoder.TypedPlayerBlockEncoder
_TypedSegmentEncoder = model_typed_encoder.TypedSegmentEncoder
_PackedScoringPlan = PackedScoringPlan
_FactorizedEvaluationResult = FactorizedEvaluationResult
_FactorizedFamilyPlan = FactorizedFamilyPlan
_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan

GLOBAL_ACTION_SPACE_SIZE = 527
_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS = PUBLIC_HEURISTIC_FRONT_ROW_SLOTS
_PUBLIC_HEURISTIC_BACK_ROW_SLOTS = PUBLIC_HEURISTIC_BACK_ROW_SLOTS
_PUBLIC_HEURISTIC_CENTER_SLOT = PUBLIC_HEURISTIC_CENTER_SLOT


_block_segments = model_typed_encoder.block_segments
_flatten_indices = model_typed_encoder.flatten_indices
_slice_by_name = slice_by_name
_header_field_index = header_field_index
_build_structured_observation_contract = build_structured_observation_contract
_bucket_card_ids = bucket_card_ids
_masked_mean_pool = masked_mean_pool
_masked_max_pool = masked_max_pool
_optional_embedding = optional_embedding
_negative_logits_fill_value = negative_logits_fill_value
_packed_row_indices = packed_row_indices
_factorized_local_row_indices = factorized_local_row_indices
_scatter_factorized_row_values = scatter_factorized_row_values
_packed_row_log_z = packed_row_log_z
_packed_local_cdf = packed_local_cdf
_uniform_from_seeds = uniform_from_seeds
_derived_sample_seeds = derived_sample_seeds
_masked_log_softmax = masked_log_softmax
_masked_entropy_from_log_probs = masked_entropy_from_log_probs


def _opponent_context_seed(policy_id: str, *, index: int) -> int:
    digest = hashlib.sha256(f"weiss_rl_opponent_context_v1\0{int(index)}\0{str(policy_id)}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _build_opponent_context_offsets(
    *,
    policy_ids: tuple[str, ...],
    hidden_size: int,
    scale: float,
) -> Tensor:
    offsets = torch.zeros((len(policy_ids) + 1, int(hidden_size)), dtype=torch.float32)
    if not policy_ids or float(scale) <= 0.0:
        return offsets
    for index, policy_id in enumerate(policy_ids, start=1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_opponent_context_seed(policy_id, index=index))
        row = torch.randn((int(hidden_size),), generator=generator, dtype=torch.float32)
        row = row / row.norm().clamp_min(1.0)
        offsets[index] = row * float(scale)
    return offsets


def _sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    return sample_masked_log_probs(
        log_probs,
        mask,
        sample_seeds=sample_seeds,
        default_index=default_index,
        temperature=temperature,
        uniform_from_seeds_fn=lambda seeds: _uniform_from_seeds(seeds, dtype=log_probs.dtype),
    )


def _sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    return sample_packed_action_scores(
        packed_scores,
        packed_ids,
        packed_offsets,
        sample_seeds,
        pass_action_id=pass_action_id,
        temperature=temperature,
        packed_row_indices_fn=_packed_row_indices,
        packed_row_log_z_fn=_packed_row_log_z,
        packed_local_cdf_fn=_packed_local_cdf,
        uniform_from_seeds_fn=lambda seeds: _uniform_from_seeds(seeds, dtype=packed_scores.dtype),
    )


class PolicyValueModel(PolicyValueModelBaseMixin, nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
        dropout_p: float | None = None,
        observation_spec: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be >= 1, got {observation_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")

        self.observation_dim = observation_dim
        self.hidden_size = config.gru_hidden_size
        self.action_dim = action_dim
        self.recurrent_core = str(config.recurrent_core).strip().lower()
        self.opponent_context_policy_ids = tuple(
            str(policy_id).strip() for policy_id in config.opponent_context_policy_ids if str(policy_id).strip()
        )
        self.opponent_context_eval_policy_ids = frozenset(
            str(policy_id).strip() for policy_id in config.opponent_context_eval_policy_ids if str(policy_id).strip()
        )
        self.opponent_context_trainable_hidden_scale = float(config.opponent_context_trainable_hidden_scale)
        self.opponent_context_trainable_recurrent_scale = float(config.opponent_context_trainable_recurrent_scale)
        self.opponent_context_trainable_action_bias_scale = float(config.opponent_context_trainable_action_bias_scale)
        self.opponent_context_trainable_candidate_residual_scale = float(
            config.opponent_context_trainable_candidate_residual_scale
        )
        self.opponent_context_candidate_residual_mode = (
            str(config.opponent_context_candidate_residual_mode).strip().lower()
        )
        self.opponent_context_candidate_residual_action_ids = tuple(
            int(action_id) for action_id in config.opponent_context_candidate_residual_action_ids
        )
        self.opponent_context_adapter_lr_multiplier = float(config.opponent_context_adapter_lr_multiplier)
        self.opponent_context_adapter_train_only = bool(config.opponent_context_adapter_train_only)
        self._opponent_context_index_by_policy_id = {
            policy_id: index for index, policy_id in enumerate(self.opponent_context_policy_ids, start=1)
        }
        self.register_buffer(
            "_opponent_context_hidden_offsets",
            _build_opponent_context_offsets(
                policy_ids=self.opponent_context_policy_ids,
                hidden_size=int(config.gru_hidden_size),
                scale=float(config.opponent_context_hidden_scale),
            ),
            persistent=False,
        )
        if self.opponent_context_policy_ids and self.opponent_context_trainable_hidden_scale > 0.0:
            self.opponent_context_hidden_adapter = nn.Parameter(
                torch.zeros((len(self.opponent_context_policy_ids) + 1, int(config.gru_hidden_size)))
            )
        if self.opponent_context_policy_ids and self.opponent_context_trainable_recurrent_scale > 0.0:
            self.opponent_context_recurrent_adapter = nn.Parameter(
                torch.zeros((len(self.opponent_context_policy_ids) + 1, int(config.gru_hidden_size)))
            )
        if self.opponent_context_policy_ids and self.opponent_context_trainable_action_bias_scale > 0.0:
            self.opponent_context_action_bias_adapter = nn.Parameter(
                torch.zeros((len(self.opponent_context_policy_ids) + 1, int(action_dim)))
            )

        encoder_dropout = config.dropout.family_a if dropout_p is None else dropout_p
        self.encoder = self._build_observation_encoder(
            observation_dim=observation_dim,
            config=config,
            observation_spec=observation_spec,
            dropout_p=encoder_dropout,
        )
        self.gru = (
            nn.GRU(input_size=config.encoder_mlp_width, hidden_size=config.gru_hidden_size, batch_first=True)
            if self.recurrent_core == "gru"
            else None
        )
        self.feedforward_core = (
            None
            if self.recurrent_core == "gru"
            else nn.Sequential(nn.Linear(config.encoder_mlp_width, config.gru_hidden_size), nn.ReLU())
        )
        self.policy_head = nn.Linear(config.gru_hidden_size, action_dim)
        self.value_head = nn.Linear(config.gru_hidden_size, 1)


class StructuredLegalPolicyValueModel(StructuredLegalPolicyValueFacadeMixin, PolicyValueModel):
    def __init__(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
        dropout_p: float | None = None,
        observation_spec: Mapping[str, Any] | None = None,
        spec_bundle: Mapping[str, Any] | None = None,
        card_table: Mapping[str, Any] | None = None,
    ) -> None:
        if spec_bundle is None:
            raise ValueError("structured_v2 encoder requires the simulator spec bundle")
        action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
        observation_contract = _build_structured_observation_contract(
            spec_bundle["observation"],
            action_catalog=action_catalog,
        )
        structured_config = replace(config, encoder_kind="typed_v1")
        super().__init__(
            observation_dim=observation_dim,
            config=structured_config,
            action_dim=action_dim,
            dropout_p=dropout_p,
            observation_spec=observation_spec,
        )
        if action_catalog.action_space_size != action_dim:
            raise ValueError(
                "structured_v2 action catalog mismatch: "
                f"expected {action_dim}, observed {action_catalog.action_space_size}"
            )
        encoder_dropout = structured_config.dropout.family_a if dropout_p is None else dropout_p
        action_feature_width = max(32, int(structured_config.encoder_mlp_width))
        self.policy_head = cast(
            Any,
            _StructuredLegalActionHead(
                latent_width=int(structured_config.gru_hidden_size),
                action_catalog=action_catalog,
                observation_contract=observation_contract,
                card_table=cached_runtime_card_table() if card_table is None else card_table,
                action_feature_width=action_feature_width,
                layer_norm=bool(structured_config.layer_norm),
                dropout_p=float(encoder_dropout),
                candidate_scoring_chunk_size=int(structured_config.candidate_scoring_chunk_size),
                cuda_learner_candidate_scoring_chunk_size=int(
                    structured_config.cuda_learner_candidate_scoring_chunk_size
                ),
                public_heuristic_logit_bias_scale=float(structured_config.public_heuristic_logit_bias_scale),
                public_heuristic_actor_logit_bias_scale=float(
                    structured_config.public_heuristic_actor_logit_bias_scale
                ),
                public_heuristic_logit_bias_families=tuple(structured_config.public_heuristic_logit_bias_families),
            ),
        )
        self.action_catalog = action_catalog
        self._structured_observation_contract = observation_contract
        self.register_buffer(
            "_card_scalar_indices",
            torch.as_tensor(observation_contract.card_scalar_indices, dtype=torch.long),
            persistent=False,
        )
        encoder_keep_mask = torch.ones((int(observation_dim),), dtype=torch.float32)
        if observation_contract.card_scalar_indices:
            encoder_keep_mask[torch.as_tensor(observation_contract.card_scalar_indices, dtype=torch.long)] = 0.0
        self.register_buffer("_encoder_input_keep_mask", encoder_keep_mask, persistent=False)
        self.supports_legal_candidate_scoring = True
        self.structured_policy_contract = str(config.structured_policy_contract).strip().lower()
        self.supports_factorized_legal_policy = self.structured_policy_contract == "factorized_v1"
        self.encoder_kind = STRUCTURED_V2_ENCODER_KIND
        self._compiled_trunk_packed_core: Any | None = None
        self._compiled_trunk_sequence_core: Any | None = None
        self._trunk_compile_last_error: str | None = None
        if self.opponent_context_policy_ids and self.opponent_context_trainable_candidate_residual_scale > 0.0:
            state_width = int(self.policy_head.state_projection[0].out_features)
            residual_width = max(1, int(config.opponent_context_candidate_residual_width))
            self.opponent_context_candidate_residual_context = nn.Parameter(
                torch.empty((len(self.opponent_context_policy_ids) + 1, residual_width))
            )
            self.opponent_context_candidate_residual_state = nn.Linear(state_width, residual_width, bias=False)
            self.opponent_context_candidate_residual_candidate = nn.Linear(state_width, residual_width, bias=False)
            self.opponent_context_candidate_residual_meta = nn.Linear(3, residual_width, bias=False)
            self.opponent_context_candidate_residual_out = nn.Linear(residual_width, 1, bias=False)
            if self.opponent_context_candidate_residual_mode in {"bilinear", "rich_bilinear"}:
                nn.init.zeros_(self.opponent_context_candidate_residual_context)
            else:
                nn.init.normal_(self.opponent_context_candidate_residual_context, mean=0.0, std=0.02)
            with torch.no_grad():
                self.opponent_context_candidate_residual_context[0].zero_()
            nn.init.zeros_(self.opponent_context_candidate_residual_out.weight)


def build_policy_value_model(
    *,
    observation_dim: int,
    config: ModelConfig,
    action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
    dropout_p: float | None = None,
    observation_spec: Mapping[str, Any] | None = None,
    spec_bundle: Mapping[str, Any] | None = None,
    card_table: Mapping[str, Any] | None = None,
) -> PolicyValueModel:
    encoder_kind = str(config.encoder_kind).strip().lower()
    if encoder_kind == STRUCTURED_V2_ENCODER_KIND:
        return StructuredLegalPolicyValueModel(
            observation_dim=observation_dim,
            config=config,
            action_dim=action_dim,
            dropout_p=dropout_p,
            observation_spec=observation_spec,
            spec_bundle=spec_bundle,
            card_table=card_table,
        )
    return PolicyValueModel(
        observation_dim=observation_dim,
        config=config,
        action_dim=action_dim,
        dropout_p=dropout_p,
        observation_spec=observation_spec,
    )


__all__ = [
    "GLOBAL_ACTION_SPACE_SIZE",
    "SEAT_COUNT",
    "STRUCTURED_V2_ENCODER_KIND",
    "ModelConfig",
    "PolicyValueModel",
    "StructuredLegalPolicyValueModel",
    "_build_mlp_stack",
    "build_policy_value_model",
]
