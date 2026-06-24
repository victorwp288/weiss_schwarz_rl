"""Stable public facade for policy/value model imports.

Implementation lives in ``weiss_rl.models``. Keep this module thin so callers
can import the model surface from one place without hiding the owner files.
"""

from __future__ import annotations

from torch import Tensor

from weiss_rl.config.models import ModelConfig
from weiss_rl.models.actions.action_plans import (
    FactorizedConditionalLogProbs,
    FactorizedEvaluationResult,
    FactorizedFamilyPlan,
    FactorizedLegalityPlan,
    PackedScoringPlan,
)
from weiss_rl.models.backbone.base import GLOBAL_ACTION_SPACE_SIZE, SEAT_COUNT, STRUCTURED_V2_ENCODER_KIND
from weiss_rl.models.backbone.layers import build_mlp_stack as _build_mlp_stack
from weiss_rl.models.backbone.tensor_ops import (
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
from weiss_rl.models.observations import typed_encoder as model_typed_encoder
from weiss_rl.models.observations.observation_contract import (
    CARD_ID_VECTOR_SLICE_NAMES,
    build_structured_observation_contract,
    header_field_index,
    slice_by_name,
)
from weiss_rl.models.policy.opponent_context import (
    build_opponent_context_offsets as _build_opponent_context_offsets,
)
from weiss_rl.models.policy.opponent_context import (
    opponent_context_seed as _opponent_context_seed,
)
from weiss_rl.models.policy.policy_value_factory import build_policy_value_model
from weiss_rl.models.policy.policy_value_model import PolicyValueModel
from weiss_rl.models.policy.structured_policy_value_model import StructuredLegalPolicyValueModel
from weiss_rl.models.public_heuristic.public_heuristic_slots import (
    PUBLIC_HEURISTIC_BACK_ROW_SLOTS,
    PUBLIC_HEURISTIC_CENTER_SLOT,
    PUBLIC_HEURISTIC_FRONT_ROW_SLOTS,
)
from weiss_rl.models.scoring.sampling import sample_masked_log_probs, sample_packed_action_scores

_CARD_ID_VECTOR_SLICE_NAMES = CARD_ID_VECTOR_SLICE_NAMES
_TypedObservationEncoder = model_typed_encoder.TypedObservationEncoder
_TypedPlayerBlockEncoder = model_typed_encoder.TypedPlayerBlockEncoder
_TypedSegmentEncoder = model_typed_encoder.TypedSegmentEncoder
_PackedScoringPlan = PackedScoringPlan
_FactorizedEvaluationResult = FactorizedEvaluationResult
_FactorizedFamilyPlan = FactorizedFamilyPlan
_FactorizedConditionalLogProbs = FactorizedConditionalLogProbs
_FactorizedLegalityPlan = FactorizedLegalityPlan

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


__all__ = [
    "GLOBAL_ACTION_SPACE_SIZE",
    "SEAT_COUNT",
    "STRUCTURED_V2_ENCODER_KIND",
    "ModelConfig",
    "PolicyValueModel",
    "StructuredLegalPolicyValueModel",
    "_build_opponent_context_offsets",
    "_build_mlp_stack",
    "_opponent_context_seed",
    "build_policy_value_model",
]
