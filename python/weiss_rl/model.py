"""Torch recurrent actor-critic model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.card_table import cached_runtime_card_table, card_feature_table
from weiss_rl.config.models import ModelConfig
from weiss_rl.eval.heuristic_public import HeuristicPublicScoringProfile, heuristic_public_scoring_profile
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.observation_layout import (
    ObservationLayout,
    ObservationPlayerBlock,
    ObservationSlice,
    parse_observation_layout,
)

GLOBAL_ACTION_SPACE_SIZE = 527
SEAT_COUNT = 2
STRUCTURED_V2_ENCODER_KIND = "structured_v2"
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


def _build_mlp_stack(
    *,
    input_dim: int,
    width: int,
    layers: int,
    layer_norm: bool,
    dropout_p: float,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"encoder input_dim must be >= 1, got {input_dim}")
    if width <= 0:
        raise ValueError(f"encoder width must be >= 1, got {width}")
    if layers <= 0:
        raise ValueError(f"encoder layers must be >= 1, got {layers}")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

    modules: list[nn.Module] = []
    in_features = input_dim
    for _ in range(layers):
        modules.append(nn.Linear(in_features, width))
        if layer_norm:
            modules.append(nn.LayerNorm(width))
        modules.append(nn.ReLU())
        if dropout_p > 0.0:
            modules.append(nn.Dropout(p=dropout_p))
        in_features = width
    return nn.Sequential(*modules)


class _TypedSegmentEncoder(nn.Module):
    def __init__(
        self,
        *,
        indices: Sequence[int],
        output_width: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if not indices:
            raise ValueError("Typed encoder segments must include at least one observation index")
        self.register_buffer("_indices", torch.as_tensor(tuple(int(index) for index in indices), dtype=torch.long))
        self._projection = _build_mlp_stack(
            input_dim=len(indices),
            width=output_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self._projection(obs.index_select(1, self._indices))


class _TypedPlayerBlockEncoder(nn.Module):
    def __init__(
        self,
        *,
        block: ObservationPlayerBlock,
        feature_width: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        slice_encoders = [
            _TypedSegmentEncoder(
                indices=current.indices,
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for current in _block_segments(block)
        ]
        self._slice_encoders = nn.ModuleList(slice_encoders)
        self._fusion = _build_mlp_stack(
            input_dim=len(slice_encoders) * feature_width,
            width=feature_width,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        encoded = [encoder(obs) for encoder in self._slice_encoders]
        return self._fusion(torch.cat(encoded, dim=1))


class _TypedObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        layout: ObservationLayout,
        feature_width: int,
        output_width: int,
        fusion_layers: int,
        layer_norm: bool,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if feature_width <= 0:
            raise ValueError(f"typed_feature_width must be >= 1, got {feature_width}")
        self._header_encoder = (
            _TypedSegmentEncoder(
                indices=tuple(field.index for field in layout.header_fields),
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            if layout.header_fields
            else None
        )
        self._player_encoders = nn.ModuleList(
            _TypedPlayerBlockEncoder(
                block=block,
                feature_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            for block in layout.player_blocks
        )
        self._tail_encoder = (
            _TypedSegmentEncoder(
                indices=_flatten_indices(layout.tail_slices),
                output_width=feature_width,
                layer_norm=layer_norm,
                dropout_p=dropout_p,
            )
            if layout.tail_slices
            else None
        )
        group_count = (
            len(self._player_encoders)
            + (0 if self._header_encoder is None else 1)
            + (0 if self._tail_encoder is None else 1)
        )
        if group_count == 0:
            raise ValueError(
                "typed_v1 encoder requires observation metadata with header_fields, player_blocks, or tail_slices"
            )
        self._fusion = _build_mlp_stack(
            input_dim=group_count * feature_width,
            width=output_width,
            layers=fusion_layers,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )

    def forward(self, obs: Tensor) -> Tensor:
        encoded_groups: list[Tensor] = []
        if self._header_encoder is not None:
            encoded_groups.append(self._header_encoder(obs))
        encoded_groups.extend(encoder(obs) for encoder in self._player_encoders)
        if self._tail_encoder is not None:
            encoded_groups.append(self._tail_encoder(obs))
        return self._fusion(torch.cat(encoded_groups, dim=1))


def _block_segments(block: ObservationPlayerBlock) -> tuple[ObservationSlice, ...]:
    if block.slices:
        return block.slices
    return (ObservationSlice(name=f"{block.name}_full", start=block.base, length=block.length),)


def _flatten_indices(slices: Sequence[ObservationSlice]) -> tuple[int, ...]:
    indices: list[int] = []
    for current in slices:
        indices.extend(current.indices)
    return tuple(indices)


_CARD_ID_VECTOR_SLICE_NAMES = frozenset(
    {
        "climax_top",
        "clock_top",
        "deck",
        "hand",
        "level_top",
        "resolution_top",
        "stock_top",
        "waiting_room_top",
    }
)


@dataclass(frozen=True, slots=True)
class _StructuredObservationContract:
    layout: ObservationLayout
    self_stage: ObservationSlice | None
    opponent_stage: ObservationSlice | None
    self_hand: ObservationSlice | None
    self_level_count: ObservationSlice | None
    self_clock_count: ObservationSlice | None
    choice_page_start_index: int | None
    choice_total_index: int | None
    stage_slot_count: int
    sentinel_hidden: int
    sentinel_empty_card: int
    card_scalar_indices: tuple[int, ...]


def _slice_by_name(block: ObservationPlayerBlock, name: str) -> ObservationSlice | None:
    for current in block.slices:
        if current.name == name:
            return current
    return None


def _header_field_index(layout: ObservationLayout, name: str) -> int | None:
    for field in layout.header_fields:
        if field.name == name:
            return int(field.index)
    return None


def _build_structured_observation_contract(
    observation_spec: Mapping[str, Any],
    *,
    action_catalog: ActionCatalog,
) -> _StructuredObservationContract:
    layout = parse_observation_layout(observation_spec)
    if not layout.self_first:
        raise ValueError("structured_v2 requires a self-first observation layout")
    if len(layout.player_blocks) < 2:
        raise ValueError("structured_v2 requires at least two player blocks in the observation layout")
    stage_slot_count = max(int(action_catalog.max_stage), 1)
    self_block = layout.player_blocks[0]
    opponent_block = layout.player_blocks[1]
    self_stage = _slice_by_name(self_block, "stage")
    opponent_stage = _slice_by_name(opponent_block, "stage")
    self_hand = _slice_by_name(self_block, "hand")
    self_level_count = _slice_by_name(self_block, "level_count")
    self_clock_count = _slice_by_name(self_block, "clock_count")

    for stage_slice, stage_name in ((self_stage, "self"), (opponent_stage, "opponent")):
        if stage_slice is None:
            continue
        if stage_slice.length % stage_slot_count != 0:
            raise ValueError(
                f"structured_v2 {stage_name} stage slice length {stage_slice.length} "
                f"is not divisible by stage slot count {stage_slot_count}"
            )

    card_scalar_indices: set[int] = set()
    for block in layout.player_blocks:
        stage_slice = _slice_by_name(block, "stage")
        if stage_slice is not None:
            slot_width = max(stage_slice.length // stage_slot_count, 1)
            for slot_index in range(stage_slot_count):
                card_scalar_indices.add(stage_slice.start + slot_index * slot_width)
        for current in block.slices:
            if current.name in _CARD_ID_VECTOR_SLICE_NAMES:
                card_scalar_indices.update(current.indices)

    return _StructuredObservationContract(
        layout=layout,
        self_stage=self_stage,
        opponent_stage=opponent_stage,
        self_hand=self_hand,
        self_level_count=self_level_count,
        self_clock_count=self_clock_count,
        choice_page_start_index=_header_field_index(layout, "choice_page_start"),
        choice_total_index=_header_field_index(layout, "choice_total"),
        stage_slot_count=stage_slot_count,
        sentinel_hidden=int(observation_spec.get("sentinel_hidden", -1)),
        sentinel_empty_card=int(observation_spec.get("sentinel_empty_card", 0)),
        card_scalar_indices=tuple(sorted(card_scalar_indices)),
    )


def _bucket_card_ids(card_ids: Tensor, *, vocab_size: int) -> Tensor:
    if vocab_size <= 1:
        return torch.zeros_like(card_ids, dtype=torch.long)
    card_ids_long = card_ids.to(dtype=torch.long)
    positive_ids = torch.where(card_ids_long > 0, card_ids_long, torch.zeros_like(card_ids_long))
    hashed = torch.remainder(positive_ids, vocab_size - 1) + 1
    return torch.where(positive_ids > 0, hashed, torch.zeros_like(hashed))


def _masked_mean_pool(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=values.dtype)
    total = (values * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return total / denom


def _masked_max_pool(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[2]))
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    pooled = masked.max(dim=1).values
    has_any = mask.any(dim=1, keepdim=True)
    return torch.where(has_any, pooled, torch.zeros_like(pooled))


def _optional_embedding(embedding: nn.Embedding, indices: Tensor) -> Tensor:
    safe_ids = torch.where(indices >= 0, indices + 1, torch.zeros_like(indices))
    return embedding(safe_ids.to(dtype=torch.long))


def _negative_logits_fill_value(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).min)
    return -1.0e9


def _packed_row_indices(offsets: Tensor) -> Tensor:
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(int(lengths.shape[0]), device=offsets.device, dtype=torch.long),
        lengths.to(dtype=torch.long),
    )


@dataclass(frozen=True, slots=True)
class _PackedScoringPlan:
    row_indices: Tensor
    family_ids: Tensor
    arg0: Tensor
    arg1: Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.family_ids.shape[0])

    def slice(self, start: int, end: int) -> "_PackedScoringPlan":
        return _PackedScoringPlan(
            row_indices=self.row_indices[start:end],
            family_ids=self.family_ids[start:end],
            arg0=self.arg0[start:end],
            arg1=self.arg1[start:end],
        )


@dataclass(frozen=True, slots=True)
class _FactorizedEvaluationResult:
    values: Tensor
    action_logp: Tensor | None
    entropy: Tensor | None
    family_log_probs: Tensor
    play_slot_log_probs: Tensor | None
    move_source_log_probs: Tensor | None
    move_slot_log_probs: Tensor | None
    attack_slot_log_probs: Tensor | None
    attack_type_log_probs: Tensor | None
    top_action_ids: Tensor | None = None
    same_family_action_logp: Tensor | None = None
    same_family_top_action_ids: Tensor | None = None


@dataclass(frozen=True, slots=True)
class _FactorizedFamilyPlan:
    row_indices: Tensor
    arg0_mask: Tensor | None
    arg1_mask: Tensor | None


@dataclass(frozen=True, slots=True)
class _FactorizedConditionalLogProbs:
    row_indices: Tensor
    log_probs: Tensor
    mask: Tensor


@dataclass(frozen=True, slots=True)
class _FactorizedLegalityPlan:
    row_count: int
    family_mask: Tensor
    family_plans: dict[int, _FactorizedFamilyPlan]


def _factorized_local_row_indices(available_rows: Tensor, selected_rows: Tensor) -> Tensor:
    if selected_rows.numel() == 0:
        return selected_rows.new_zeros((0,), dtype=torch.long)
    if available_rows.numel() == 0:
        raise ValueError("factorized row lookup requires at least one available row")
    positions = torch.searchsorted(available_rows, selected_rows)
    if bool((positions >= available_rows.shape[0]).any().item()):
        raise ValueError("factorized row lookup exceeded available rows")
    matched_rows = available_rows.index_select(0, positions)
    if not bool(torch.equal(matched_rows, selected_rows)):
        raise ValueError("factorized row lookup requires selected rows to be legal for the chosen family")
    return positions


def _scatter_factorized_row_values(
    row_count: int,
    row_indices: Tensor,
    values: Tensor,
    *,
    fill_value: float = -torch.inf,
) -> Tensor:
    output = values.new_full((row_count, *values.shape[1:]), fill_value)
    if row_indices.numel() > 0:
        output.index_copy_(0, row_indices.to(dtype=torch.long), values)
    return output


def _packed_row_log_z(scores: Tensor, offsets: Tensor) -> Tensor:
    row_count = int(offsets.shape[0] - 1)
    if row_count < 0:
        raise ValueError("packed offsets must contain at least one row boundary")
    row_log_z = torch.full((row_count,), -torch.inf, device=scores.device, dtype=scores.dtype)
    if scores.numel() == 0 or row_count == 0:
        return row_log_z
    lengths = offsets[1:] - offsets[:-1]
    non_empty_rows = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return row_log_z
    non_empty_lengths = lengths[non_empty_rows].to(dtype=torch.long)
    segment_max = torch.segment_reduce(scores, reduce="max", lengths=non_empty_lengths)
    repeated_max = torch.repeat_interleave(segment_max, non_empty_lengths)
    shifted = scores - repeated_max
    exp_shifted = torch.exp(shifted)
    segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_lengths)
    row_log_z[non_empty_rows] = torch.log(segment_sum) + segment_max
    return row_log_z


def _packed_local_cdf(probabilities: Tensor, offsets: Tensor) -> Tensor:
    if probabilities.numel() == 0:
        return probabilities
    row_count = int(offsets.shape[0] - 1)
    row_indices = _packed_row_indices(offsets)
    cumulative = torch.cumsum(probabilities, dim=0)
    base = torch.zeros((row_count,), dtype=probabilities.dtype, device=probabilities.device)
    if row_count > 1:
        starts = offsets[1:-1].to(dtype=torch.long)
        base[1:] = cumulative.index_select(0, starts - 1)
    return cumulative - base.index_select(0, row_indices)


def _uniform_from_seeds(sample_seeds: Tensor, *, dtype: torch.dtype) -> Tensor:
    seed_float = sample_seeds.to(dtype=torch.float64)
    hashed = torch.sin(seed_float * 12.9898 + 78.233) * 43758.5453123
    uniform = torch.frac(hashed).to(dtype=dtype)
    eps = torch.finfo(dtype).eps
    return torch.clamp(uniform, min=eps, max=1.0 - eps)


def _derived_sample_seeds(sample_seeds: Tensor, *, salt: int) -> Tensor:
    mixed = sample_seeds.to(dtype=torch.long)
    return mixed ^ torch.full_like(mixed, int(salt), dtype=torch.long)


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("masked_log_softmax requires logits and mask with matching shapes")
    negative_fill = torch.full_like(logits, _negative_logits_fill_value(logits.dtype))
    masked_logits = torch.where(mask, logits, negative_fill)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return torch.where(mask, log_probs, negative_fill)


def _masked_entropy_from_log_probs(log_probs: Tensor, mask: Tensor) -> Tensor:
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log_probs).sum(dim=-1)


def _sample_masked_log_probs(
    log_probs: Tensor,
    mask: Tensor,
    *,
    sample_seeds: Tensor,
    default_index: int = 0,
) -> tuple[Tensor, Tensor]:
    if log_probs.ndim != 2 or mask.ndim != 2 or log_probs.shape != mask.shape:
        raise ValueError("sampled masked log_probs requires 2D tensors with matching shape")
    row_count = int(log_probs.shape[0])
    if sample_seeds.ndim != 1 or int(sample_seeds.shape[0]) != row_count:
        raise ValueError(f"sample_seeds must have shape ({row_count},)")
    actions = torch.full((row_count,), int(default_index), device=log_probs.device, dtype=torch.long)
    selected_logp = torch.zeros((row_count,), device=log_probs.device, dtype=log_probs.dtype)
    if row_count == 0:
        return actions, selected_logp
    row_has_candidates = mask.any(dim=1)
    non_empty_rows = torch.nonzero(row_has_candidates, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return actions, selected_logp
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    cdf = torch.cumsum(probs, dim=1)
    thresholds = _uniform_from_seeds(
        sample_seeds.index_select(0, non_empty_rows).to(device=log_probs.device, dtype=torch.long),
        dtype=log_probs.dtype,
    ).unsqueeze(1)
    cdf_rows = cdf.index_select(0, non_empty_rows)
    chosen = cdf_rows >= thresholds
    chosen_indices = chosen.to(dtype=torch.int64).argmax(dim=1)
    fallback_indices = mask.index_select(0, non_empty_rows).to(dtype=torch.int64).argmax(dim=1)
    chosen_indices = torch.where(chosen.any(dim=1), chosen_indices, fallback_indices)
    actions[non_empty_rows] = chosen_indices
    selected_logp[non_empty_rows] = (
        log_probs.index_select(0, non_empty_rows)
        .gather(
            1,
            chosen_indices.unsqueeze(1),
        )
        .squeeze(1)
    )
    return actions, selected_logp


def _sample_packed_action_scores(
    packed_scores: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    sample_seeds: Tensor,
    *,
    pass_action_id: int,
) -> tuple[Tensor, Tensor]:
    if packed_scores.ndim != 1:
        raise ValueError("packed_scores must be 1D")
    if packed_ids.ndim != 1 or packed_offsets.ndim != 1:
        raise ValueError("packed ids and offsets must be 1D")
    row_count = int(packed_offsets.shape[0] - 1)
    if sample_seeds.ndim != 1 or int(sample_seeds.shape[0]) != row_count:
        raise ValueError(f"sample_seeds must have shape ({row_count},)")
    if int(packed_offsets[0].item()) != 0 or int(packed_offsets[-1].item()) != int(packed_scores.shape[0]):
        raise ValueError("packed offsets must describe the packed score vector exactly")

    lengths = packed_offsets[1:] - packed_offsets[:-1]
    actions = torch.full(
        (row_count,),
        int(pass_action_id),
        device=packed_scores.device,
        dtype=torch.long,
    )
    selected_logp = torch.zeros((row_count,), device=packed_scores.device, dtype=packed_scores.dtype)
    non_empty_rows = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() == 0:
        return actions, selected_logp

    non_empty_lengths = lengths[non_empty_rows].to(dtype=torch.long)
    row_indices = _packed_row_indices(packed_offsets)
    row_log_z = _packed_row_log_z(packed_scores, packed_offsets)
    repeated_log_z = row_log_z.index_select(0, row_indices)
    log_probs = packed_scores - repeated_log_z
    probs = torch.exp(log_probs)
    local_cdf = _packed_local_cdf(probs, packed_offsets)
    thresholds = _uniform_from_seeds(
        sample_seeds.to(device=packed_scores.device, dtype=torch.long).index_select(0, non_empty_rows),
        dtype=packed_scores.dtype,
    )
    repeated_thresholds = thresholds.index_select(0, row_indices)
    previous_cdf = local_cdf - probs
    chosen = (local_cdf >= repeated_thresholds) & (previous_cdf < repeated_thresholds)
    packed_positions = torch.arange(packed_scores.shape[0], device=packed_scores.device, dtype=packed_scores.dtype)
    sentinel = torch.full_like(packed_positions, float(packed_scores.shape[0]))
    chosen_positions = torch.segment_reduce(
        torch.where(chosen, packed_positions, sentinel),
        reduce="amin",
        lengths=non_empty_lengths,
    ).to(dtype=torch.long)
    missing_rows = torch.nonzero(chosen_positions == packed_scores.shape[0], as_tuple=False).squeeze(1)
    if missing_rows.numel() > 0:
        fallback_positions = (
            packed_offsets[1:]
            .to(device=packed_scores.device, dtype=torch.long)
            .index_select(0, non_empty_rows.index_select(0, missing_rows))
            - 1
        )
        chosen_positions = chosen_positions.clone()
        chosen_positions[missing_rows] = fallback_positions
    chosen_actions = packed_ids.index_select(0, chosen_positions)
    chosen_logp = log_probs.index_select(0, chosen_positions)
    actions[non_empty_rows] = chosen_actions
    selected_logp[non_empty_rows] = chosen_logp
    return actions, selected_logp


class PolicyValueModel(nn.Module):
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

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return torch.zeros(batch_size, self.hidden_size, device=hidden_device, dtype=hidden_dtype)

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return torch.zeros(batch_size, SEAT_COUNT, self.hidden_size, device=hidden_device, dtype=hidden_dtype)

    def encode(self, obs: Tensor) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        return self.encoder(obs_batch)

    def recurrent_step(self, encoded_obs: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        hidden_batch = self._prepare_hidden_state(hidden_state, batch_size=batch_size, like=encoded_obs)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            recurrent_output, next_hidden = self.gru(encoded_obs.unsqueeze(1), hidden_batch.unsqueeze(0))
            return recurrent_output[:, 0, :], next_hidden[0]
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), hidden_batch

    def recurrent_step_seat_aware(
        self,
        encoded_obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        seat_hidden_batch = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded_obs,
        )
        acting_seat_batch = self._prepare_acting_seat(acting_seat, batch_size=batch_size, device=encoded_obs.device)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            acting_hidden_batch = self._select_acting_hidden(seat_hidden_batch, acting_seat_batch)
            recurrent_output, next_acting_hidden = self.gru(encoded_obs.unsqueeze(1), acting_hidden_batch.unsqueeze(0))
            next_seat_hidden = self._write_acting_hidden(seat_hidden_batch, acting_seat_batch, next_acting_hidden[0])
            return recurrent_output[:, 0, :], next_seat_hidden
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), seat_hidden_batch

    def recurrent_step_seat_aware_inplace(
        self,
        encoded_obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoded_obs.ndim != 2:
            raise ValueError(f"encoded_obs must be 2D (batch, latent), got shape {tuple(encoded_obs.shape)}")

        batch_size = encoded_obs.shape[0]
        seat_hidden_batch = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded_obs,
        )
        acting_seat_batch = self._prepare_acting_seat(acting_seat, batch_size=batch_size, device=encoded_obs.device)
        if self.recurrent_core == "gru":
            assert self.gru is not None
            acting_hidden_batch = self._select_acting_hidden(seat_hidden_batch, acting_seat_batch)
            recurrent_output, next_acting_hidden = self.gru(encoded_obs.unsqueeze(1), acting_hidden_batch.unsqueeze(0))
            next_hidden = next_acting_hidden[0]
            if next_hidden.dtype != seat_hidden_batch.dtype:
                next_hidden = next_hidden.to(dtype=seat_hidden_batch.dtype)
            batch_index = torch.arange(seat_hidden_batch.shape[0], device=seat_hidden_batch.device)
            seat_hidden_batch[batch_index, acting_seat_batch] = next_hidden
            return recurrent_output[:, 0, :], seat_hidden_batch
        assert self.feedforward_core is not None
        return self.feedforward_core(encoded_obs), seat_hidden_batch

    def forward(
        self,
        obs: Tensor,
        hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def advance_seat_hidden(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        encoded_obs = self.encode(obs)
        _, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return next_seat_hidden

    def value_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        encoded_obs = self.encode(obs)
        recurrent_output, _next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return self.value_head(recurrent_output).squeeze(-1)

    def forward_sequence_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if legal_actions is not None:
            raise ValueError("forward_sequence_seat_aware with legal_actions is only supported on structured models")
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        batch_size = int(obs.shape[1])
        seat_hidden = self._prepare_seat_hidden_state(seat_hidden_state, batch_size=batch_size, like=obs[0])
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []
        for step_obs, step_seat in zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True):
            step_logits, step_value, seat_hidden = self.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden,
            )
            logits_steps.append(step_logits)
            value_steps.append(step_value)
        return torch.stack(logits_steps, dim=0), torch.stack(value_steps, dim=0), seat_hidden

    def forward_sequence_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise ValueError("forward_sequence_packed_seat_aware is only supported on structured models")

    def forward_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise ValueError("forward_packed_seat_aware is only supported on structured models")

    def sample_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise ValueError("sample_packed_seat_aware is only supported on structured models")

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> PolicyValueModel:
        return self

    def _build_observation_encoder(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        observation_spec: Mapping[str, Any] | None,
        dropout_p: float,
    ) -> nn.Module:
        encoder_kind = str(config.encoder_kind).strip().lower()
        if encoder_kind == "mlp":
            return _build_mlp_stack(
                input_dim=observation_dim,
                width=config.encoder_mlp_width,
                layers=config.encoder_mlp_layers,
                layer_norm=config.layer_norm,
                dropout_p=dropout_p,
            )
        if encoder_kind not in {"typed_v1", STRUCTURED_V2_ENCODER_KIND}:
            raise ValueError(f"Unsupported model.encoder_kind: {config.encoder_kind!r}")
        if observation_spec is None:
            raise ValueError(f"{encoder_kind} encoder requires observation_spec from the simulator spec bundle")
        layout = parse_observation_layout(observation_spec)
        if layout.obs_len != observation_dim:
            raise ValueError(
                f"{encoder_kind} observation spec length mismatch: "
                f"expected {observation_dim}, observed {layout.obs_len}"
            )
        return _TypedObservationEncoder(
            layout=layout,
            feature_width=config.typed_feature_width,
            output_width=config.encoder_mlp_width,
            fusion_layers=config.encoder_mlp_layers,
            layer_norm=config.layer_norm,
            dropout_p=dropout_p,
        )

    def _require_observation_batch(self, obs: Tensor) -> Tensor:
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2D (batch, observation), got shape {tuple(obs.shape)}")
        if obs.shape[1] != self.observation_dim:
            raise ValueError(f"obs feature dimension mismatch: expected {self.observation_dim}, got {obs.shape[1]}")
        return obs.to(dtype=self._reference_parameter().dtype)

    def _hidden_tensor_device_dtype(
        self,
        *,
        batch_size: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        reference = self._reference_parameter()
        hidden_device: torch.device = reference.device if device is None else device
        hidden_dtype: torch.dtype = reference.dtype if dtype is None else dtype
        return hidden_device, hidden_dtype

    def _prepare_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        if hidden_state is None:
            return self.initial_hidden(batch_size, device=like.device, dtype=like.dtype)
        if hidden_state.ndim != 2:
            raise ValueError(f"hidden_state must be 2D (batch, hidden_size), got shape {tuple(hidden_state.shape)}")
        if hidden_state.shape[0] != batch_size:
            raise ValueError(f"hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
        if hidden_state.shape[1] != self.hidden_size:
            raise ValueError(f"hidden_state feature mismatch: expected {self.hidden_size}, got {hidden_state.shape[1]}")
        return hidden_state.to(device=like.device, dtype=like.dtype)

    def _prepare_seat_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        if hidden_state is None:
            return self.initial_seat_hidden(batch_size, device=like.device, dtype=like.dtype)
        if hidden_state.ndim != 3:
            raise ValueError(
                f"seat_hidden_state must be 3D (batch, seat, hidden_size), got shape {tuple(hidden_state.shape)}"
            )
        if hidden_state.shape[0] != batch_size:
            raise ValueError(f"seat_hidden_state batch mismatch: expected {batch_size}, got {hidden_state.shape[0]}")
        if hidden_state.shape[1] != SEAT_COUNT:
            raise ValueError(f"seat_hidden_state seat mismatch: expected {SEAT_COUNT}, got {hidden_state.shape[1]}")
        if hidden_state.shape[2] != self.hidden_size:
            raise ValueError(
                f"seat_hidden_state feature mismatch: expected {self.hidden_size}, got {hidden_state.shape[2]}"
            )
        return hidden_state.to(device=like.device, dtype=like.dtype)

    def _prepare_acting_seat(self, acting_seat: int | Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        if isinstance(acting_seat, int):
            seat_batch = torch.full((batch_size,), acting_seat, device=device, dtype=torch.long)
        else:
            if acting_seat.is_floating_point() or acting_seat.is_complex():
                raise ValueError("acting_seat must contain integer seat ids")
            if acting_seat.ndim == 0:
                seat_batch = acting_seat.to(device=device, dtype=torch.long).expand(batch_size)
            elif acting_seat.ndim == 1:
                if acting_seat.shape[0] != batch_size:
                    raise ValueError(f"acting_seat batch mismatch: expected {batch_size}, got {acting_seat.shape[0]}")
                seat_batch = acting_seat.to(device=device, dtype=torch.long)
            else:
                raise ValueError(f"acting_seat must be scalar or 1D [batch], got shape {tuple(acting_seat.shape)}")
        if not torch.all((seat_batch == 0) | (seat_batch == 1)):
            raise ValueError("acting_seat values must be 0 or 1")
        return seat_batch

    def _select_acting_hidden(self, seat_hidden_state: Tensor, acting_seat: Tensor) -> Tensor:
        acting_index = acting_seat.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        return torch.gather(seat_hidden_state, dim=1, index=acting_index).squeeze(1)

    def _write_acting_hidden(
        self,
        seat_hidden_state: Tensor,
        acting_seat: Tensor,
        next_acting_hidden: Tensor,
    ) -> Tensor:
        next_seat_hidden = seat_hidden_state.clone()
        if next_acting_hidden.dtype != next_seat_hidden.dtype:
            next_acting_hidden = next_acting_hidden.to(dtype=next_seat_hidden.dtype)
        batch_index = torch.arange(seat_hidden_state.shape[0], device=seat_hidden_state.device)
        next_seat_hidden[batch_index, acting_seat] = next_acting_hidden
        return next_seat_hidden

    def _reference_parameter(self) -> Tensor:
        try:
            return next(self.parameters())
        except StopIteration as exc:
            raise RuntimeError("Model has no parameters to use as a reference tensor") from exc


class _StructuredLegalActionHead(nn.Module):
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
        lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(latent.shape[0], device=latent.device, dtype=torch.long),
            lengths,
        )
        return self._score_candidates_chunked(
            resolved_state_repr,
            row_indices,
            ids,
            resolved_context,
            candidate_meta=meta,
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

    def _slot_preference_values(self, slot_indices: Tensor, *, dtype: torch.dtype) -> Tensor:
        if self._public_slot_preference.numel() == 0:
            return slot_indices.new_zeros(slot_indices.shape, dtype=dtype)
        valid = (slot_indices >= 0) & (slot_indices < int(self._public_slot_preference.shape[0]))
        safe_slots = torch.where(valid, slot_indices, torch.zeros_like(slot_indices)).to(dtype=torch.long)
        values = self._public_slot_preference.index_select(0, safe_slots).to(dtype=dtype)
        return values * valid.to(dtype=dtype)

    def _public_prefer_lower(self, values: Tensor, *, dtype: torch.dtype) -> Tensor:
        return torch.where(values >= 0, -values.to(dtype=dtype), values.new_zeros(values.shape, dtype=dtype))

    def _public_slot_action_score(
        self,
        slot_values: Tensor,
        slot_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        power = torch.clamp(slot_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
        return self._slot_preference_values(slot_values, dtype=dtype) + torch.floor(power / 1000.0)

    def _combine_public_heuristic_scores(
        self,
        score0: Tensor,
        score1: Tensor,
        score2: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        return score0.to(dtype=dtype) * 32.0 + score1.to(dtype=dtype) + (score2.to(dtype=dtype) / 4.0)

    def _public_heuristic_logit_bias_scale_for(self, scoring_mode: str) -> float:
        resolved_mode = self._resolve_scoring_mode(scoring_mode)
        if resolved_mode == "actor":
            return float(self._public_heuristic_actor_logit_bias_scale)
        return float(self._public_heuristic_logit_bias_scale)

    def _apply_public_heuristic_bias(
        self,
        scores: Tensor,
        raw_scores: Tensor,
        *,
        scale: float,
        family_ids: Tensor | None = None,
    ) -> Tensor:
        if scale <= 0.0 or raw_scores.numel() == 0:
            return scores
        bias = raw_scores.to(dtype=scores.dtype) * (float(scale) / 100.0)
        if family_ids is None or self._public_heuristic_bias_family_ids.numel() == 0:
            return scores + bias
        allowed = torch.isin(
            family_ids.to(device=self._public_heuristic_bias_family_ids.device, dtype=torch.long),
            self._public_heuristic_bias_family_ids,
        ).to(device=scores.device, dtype=scores.dtype)
        return scores + (bias * allowed)

    def _play_public_heuristic_raw(
        self,
        stage_slots: Tensor,
        target_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        slot_pref = self._slot_preference_values(stage_slots, dtype=dtype)
        front_bonus = torch.where(
            stage_slots < len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS),
            stage_slots.new_full(stage_slots.shape, 40.0, dtype=dtype),
            torch.where(
                stage_slots < len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS) + len(_PUBLIC_HEURISTIC_BACK_ROW_SLOTS),
                stage_slots.new_full(stage_slots.shape, 20.0, dtype=dtype),
                stage_slots.new_zeros(stage_slots.shape, dtype=dtype),
            ),
        )
        occupied = target_numeric[:, 0].to(dtype=dtype) > 0.5
        raw = stage_slots.new_full(stage_slots.shape, 650.0, dtype=dtype) + slot_pref + front_bonus
        return torch.where(occupied, stage_slots.new_full(stage_slots.shape, -1000.0, dtype=dtype), raw)

    def _move_public_heuristic_raw(
        self,
        from_slots: Tensor,
        to_slots: Tensor,
        source_numeric: Tensor,
        target_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        source_pref = self._slot_preference_values(from_slots, dtype=dtype)
        target_pref = self._slot_preference_values(to_slots, dtype=dtype)
        improvement = target_pref - source_pref
        front_row_threshold = len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS)
        back_to_front = (from_slots >= front_row_threshold) & (to_slots < front_row_threshold)
        move_to_center = (to_slots == _PUBLIC_HEURISTIC_CENTER_SLOT) & (from_slots != _PUBLIC_HEURISTIC_CENTER_SLOT)
        bonus = back_to_front.to(dtype=dtype) * 30.0 + move_to_center.to(dtype=dtype) * 15.0
        valid = (source_numeric[:, 0].to(dtype=dtype) > 0.5) & (target_numeric[:, 0].to(dtype=dtype) <= 0.5)
        raw = from_slots.new_full(from_slots.shape, 120.0, dtype=dtype) + improvement + bonus
        return torch.where(valid, raw, from_slots.new_full(from_slots.shape, -1000.0, dtype=dtype))

    def _attack_public_heuristic_raw(
        self,
        slot_values: Tensor,
        attack_type_values: Tensor,
        source_numeric: Tensor,
        defender_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        slot_pref = self._slot_preference_values(slot_values, dtype=dtype)
        attacker_occupied = source_numeric[:, 0].to(dtype=dtype) > 0.5
        attacker_power = source_numeric[:, 3].to(dtype=dtype)
        attacker_effective_soul = source_numeric[:, 5].to(dtype=dtype)
        side_attack_allowed = source_numeric[:, 6].to(dtype=dtype) > 0.5
        defender_occupied = defender_numeric[:, 0].to(dtype=dtype) > 0.5
        defender_power = defender_numeric[:, 3].to(dtype=dtype)
        attack_type_score = slot_values.new_zeros(slot_values.shape, dtype=dtype)
        direct_mask = attack_type_values == 2
        frontal_mask = attack_type_values == 0
        side_mask = attack_type_values == 1
        attack_type_score = torch.where(
            direct_mask,
            torch.where(
                defender_occupied,
                slot_values.new_full(slot_values.shape, 15.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 60.0, dtype=dtype),
            ),
            attack_type_score,
        )
        attack_type_score = torch.where(
            frontal_mask,
            torch.where(
                attacker_power >= defender_power,
                slot_values.new_full(slot_values.shape, 45.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 25.0, dtype=dtype),
            ),
            attack_type_score,
        )
        attack_type_score = torch.where(
            side_mask,
            torch.where(
                side_attack_allowed,
                slot_values.new_full(slot_values.shape, 40.0, dtype=dtype),
                slot_values.new_full(slot_values.shape, 5.0, dtype=dtype),
            ),
            attack_type_score,
        )
        power_term = attacker_power * 20.0
        soul_term = attacker_effective_soul * 16.0
        raw = (
            slot_values.new_full(slot_values.shape, 900.0, dtype=dtype)
            + attack_type_score
            + slot_pref
            + power_term
            + soul_term
        )
        return torch.where(attacker_occupied, raw, slot_values.new_full(slot_values.shape, -1000.0, dtype=dtype))

    def _slot_family_public_heuristic_raw(
        self,
        family_ids: Tensor,
        slot_values: Tensor,
        slot_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        slot_pref = self._slot_preference_values(slot_values, dtype=dtype)
        power_term = slot_numeric[:, 3].to(dtype=dtype) * 20.0
        raw = slot_values.new_zeros(slot_values.shape, dtype=dtype)
        if self._encore_pay_family_id >= 0:
            raw = torch.where(
                family_ids == self._encore_pay_family_id,
                slot_values.new_full(slot_values.shape, 700.0, dtype=dtype) + slot_pref + power_term,
                raw,
            )
        if self._encore_decline_family_id >= 0:
            raw = torch.where(
                family_ids == self._encore_decline_family_id,
                slot_values.new_full(slot_values.shape, 110.0, dtype=dtype) + slot_pref + power_term,
                raw,
            )
        return raw

    def _public_attack_profile(
        self,
        self_stage_numeric: Tensor,
        opponent_stage_numeric: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        front_row_count = len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS)
        attackers_available = (
            (
                (self_stage_numeric[:, :front_row_count, 0].to(dtype=dtype) > 0.5)
                & ~(self_stage_numeric[:, :front_row_count, 2].to(dtype=dtype) > 0.5)
            )
            .sum(dim=1)
            .to(dtype=dtype)
        )
        front_defenders = (
            (opponent_stage_numeric[:, :front_row_count, 0].to(dtype=dtype) > 0.5).sum(dim=1).to(dtype=dtype)
        )
        return attackers_available, front_defenders

    def _hand_public_heuristic_raw(
        self,
        family_ids: Tensor,
        hand_indices: Tensor,
        *,
        attackers_available: Tensor,
        front_defenders: Tensor,
        self_level_count: Tensor,
        self_clock_count: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        raw = hand_indices.new_zeros(hand_indices.shape, dtype=dtype)
        lower_index_bonus = self._public_prefer_lower(hand_indices, dtype=dtype)
        if self._climax_play_family_id >= 0:
            climax_bonus = (
                attackers_available * 10.0
                + front_defenders * 4.0
                + torch.where(
                    attackers_available > 0.0,
                    hand_indices.new_full(hand_indices.shape, 10.0, dtype=dtype),
                    hand_indices.new_full(hand_indices.shape, -20.0, dtype=dtype),
                )
            )
            raw = torch.where(
                family_ids == self._climax_play_family_id,
                hand_indices.new_full(hand_indices.shape, 550.0, dtype=dtype) + climax_bonus + lower_index_bonus,
                raw,
            )
        if self._clock_from_hand_family_id >= 0:
            clock_bonus = torch.where(
                (self_level_count <= 0.0) & (self_clock_count < 6.0),
                40.0 - self_clock_count,
                self_clock_count.new_full(self_clock_count.shape, 10.0, dtype=dtype),
            )
            raw = torch.where(
                family_ids == self._clock_from_hand_family_id,
                hand_indices.new_full(hand_indices.shape, 500.0, dtype=dtype) + clock_bonus + lower_index_bonus,
                raw,
            )
        if self._main_event_family_id >= 0:
            raw = torch.where(
                family_ids == self._main_event_family_id,
                hand_indices.new_full(hand_indices.shape, 330.0, dtype=dtype) + lower_index_bonus,
                raw,
            )
        if self._mulligan_select_family_id >= 0:
            raw = torch.where(
                family_ids == self._mulligan_select_family_id,
                hand_indices.new_full(hand_indices.shape, 120.0, dtype=dtype) + lower_index_bonus,
                raw,
            )
        return raw

    def _index_public_heuristic_raw(
        self,
        family_ids: Tensor,
        index_values: Tensor,
        *,
        choice_page_start: Tensor,
        choice_total: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        raw = index_values.new_zeros(index_values.shape, dtype=dtype)
        lower_index_bonus = self._public_prefer_lower(index_values, dtype=dtype)
        if self._choice_select_family_id >= 0:
            raw = torch.where(
                family_ids == self._choice_select_family_id,
                index_values.new_full(index_values.shape, 300.0, dtype=dtype) + lower_index_bonus,
                raw,
            )
        if self._level_up_family_id >= 0:
            raw = torch.where(
                family_ids == self._level_up_family_id,
                index_values.new_full(index_values.shape, 290.0, dtype=dtype) + lower_index_bonus,
                raw,
            )
        if self._trigger_order_family_id >= 0:
            raw = torch.where(
                family_ids == self._trigger_order_family_id,
                index_values.new_full(index_values.shape, 280.0, dtype=dtype) + lower_index_bonus,
                raw,
            )
        if self._next_page_family_id >= 0:
            raw = torch.where(
                family_ids == self._next_page_family_id,
                index_values.new_full(index_values.shape, 170.0, dtype=dtype)
                + torch.clamp(choice_total - (choice_page_start + 16.0), min=0.0),
                raw,
            )
        if self._prev_page_family_id >= 0:
            raw = torch.where(
                family_ids == self._prev_page_family_id,
                index_values.new_full(index_values.shape, 170.0, dtype=dtype) + torch.clamp(choice_page_start, min=0.0),
                raw,
            )
        return raw

    def _default_public_heuristic_raw(
        self,
        family_ids: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        raw = family_ids.new_zeros(family_ids.shape, dtype=dtype)
        if self._mulligan_confirm_family_id >= 0:
            raw = torch.where(
                family_ids == self._mulligan_confirm_family_id,
                family_ids.new_full(family_ids.shape, 260.0, dtype=dtype),
                raw,
            )
        if self._pass_family_id >= 0:
            raw = torch.where(
                family_ids == self._pass_family_id,
                family_ids.new_full(family_ids.shape, 160.0, dtype=dtype),
                raw,
            )
        return raw

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

    def _score_packed_public_heuristic_chunked(
        self,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> Tensor:
        if scoring_plan.candidate_count == 0:
            return torch.zeros((0,), device=scoring_plan.row_indices.device, dtype=dtype)
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            scores_chunks.append(
                self._score_packed_public_heuristic_plan(
                    scoring_plan.slice(start, end),
                    observation_context,
                    dtype=dtype,
                    scoring_profile=scoring_profile,
                )
            )
        return torch.cat(scores_chunks, dim=0)

    def _score_packed_public_heuristic_plan(
        self,
        scoring_plan: _PackedScoringPlan,
        observation_context: Mapping[str, Tensor],
        *,
        dtype: torch.dtype,
        scoring_profile: HeuristicPublicScoringProfile,
    ) -> Tensor:
        row_indices_long = scoring_plan.row_indices.to(dtype=torch.long)
        candidate_count = scoring_plan.candidate_count
        score0 = torch.full((candidate_count,), -1000.0, dtype=dtype, device=row_indices_long.device)
        score1 = torch.zeros((candidate_count,), dtype=dtype, device=row_indices_long.device)
        score2 = torch.zeros((candidate_count,), dtype=dtype, device=row_indices_long.device)

        self_stage_numeric = observation_context["self_stage_numeric"]
        opponent_stage_numeric = observation_context["opponent_stage_numeric"]
        self_level_count = observation_context["self_level_count"].to(device=row_indices_long.device, dtype=dtype)
        self_clock_count = observation_context["self_clock_count"].to(device=row_indices_long.device, dtype=dtype)
        choice_page_start = observation_context["choice_page_start"].to(device=row_indices_long.device, dtype=dtype)
        choice_total = observation_context["choice_total"].to(device=row_indices_long.device, dtype=dtype)

        attackers_available = (
            (
                (self_stage_numeric[:, : len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS), 0].to(dtype=dtype) > 0.5)
                & ~(self_stage_numeric[:, : len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS), 2].to(dtype=dtype) > 0.5)
            )
            .sum(dim=1)
            .to(dtype=dtype)
        )
        front_defenders = (
            (opponent_stage_numeric[:, : len(_PUBLIC_HEURISTIC_FRONT_ROW_SLOTS), 0].to(dtype=dtype) > 0.5)
            .sum(dim=1)
            .to(dtype=dtype)
        )

        (
            play_indices,
            hand_indices,
            move_indices,
            attack_indices,
            slot_family_indices,
            index_family_indices,
            default_indices,
        ) = self._partition_candidate_family_indices(scoring_plan.family_ids)

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
            attack_source_context, attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            del attack_source_context
            _defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                opponent_stage_numeric,
                attack_rows,
                attack_slot_values,
            )
            del _defender_context
            slot_pref = self._slot_preference_values(attack_slot_values, dtype=dtype)
            attacker_power = torch.clamp(attack_source_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
            attacker_soul = torch.clamp(attack_source_numeric[:, 5].to(dtype=dtype) * 4.0, min=0.0)
            defender_power = torch.clamp(defender_numeric[:, 3].to(dtype=dtype) * 20000.0, min=0.0)
            attacker_occupied = attack_source_numeric[:, 0].to(dtype=dtype) > 0.5
            defender_occupied = defender_numeric[:, 0].to(dtype=dtype) > 0.5
            side_attack_allowed = attack_source_numeric[:, 6].to(dtype=dtype) > 0.5
            type_score = torch.zeros(attack_slot_values.shape, dtype=dtype, device=row_indices_long.device)
            if self._direct_attack_type_id >= 0:
                direct = attack_type_values == self._direct_attack_type_id
                type_score = torch.where(
                    direct,
                    torch.where(
                        defender_occupied,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_direct_blocked_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_direct_open_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            if self._frontal_attack_type_id >= 0:
                frontal = attack_type_values == self._frontal_attack_type_id
                type_score = torch.where(
                    frontal,
                    torch.where(
                        attacker_power >= defender_power,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_frontal_win_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_frontal_loss_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            if self._side_attack_type_id >= 0:
                side = attack_type_values == self._side_attack_type_id
                type_score = torch.where(
                    side,
                    torch.where(
                        side_attack_allowed,
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_side_allowed_bonus),
                            dtype=dtype,
                        ),
                        attack_slot_values.new_full(
                            attack_slot_values.shape,
                            float(scoring_profile.attack_side_blocked_bonus),
                            dtype=dtype,
                        ),
                    ),
                    type_score,
                )
            attack_score = (
                type_score
                + slot_pref
                + (attacker_soul * float(scoring_profile.attack_soul_scale))
                + torch.floor(attacker_power / 1000.0)
            )
            attack_score = torch.where(
                attacker_occupied,
                attack_score,
                attack_slot_values.new_full(attack_slot_values.shape, -1000.0, dtype=dtype),
            )
            score0.index_fill_(0, attack_indices, float(scoring_profile.attack_priority))
            score1.index_copy_(0, attack_indices, attack_score)

        if slot_family_indices.numel() > 0:
            slot_rows = row_indices_long.index_select(0, slot_family_indices)
            slot_family_ids = scoring_plan.family_ids.index_select(0, slot_family_indices)
            slot_values = scoring_plan.arg0.index_select(0, slot_family_indices)
            _slot_context, slot_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                slot_rows,
                slot_values,
            )
            del _slot_context
            slot_scores = self._public_slot_action_score(slot_values, slot_numeric, dtype=dtype)
            if self._encore_pay_family_id >= 0:
                encore_pay_mask = slot_family_ids == self._encore_pay_family_id
                if bool(encore_pay_mask.any().item()):
                    pay_indices = slot_family_indices[encore_pay_mask]
                    score0.index_fill_(0, pay_indices, float(scoring_profile.encore_pay_priority))
                    score1.index_copy_(0, pay_indices, slot_scores[encore_pay_mask])
            if self._encore_decline_family_id >= 0:
                encore_decline_mask = slot_family_ids == self._encore_decline_family_id
                if bool(encore_decline_mask.any().item()):
                    decline_indices = slot_family_indices[encore_decline_mask]
                    score0.index_fill_(0, decline_indices, float(scoring_profile.encore_decline_priority))
                    score1.index_copy_(0, decline_indices, slot_scores[encore_decline_mask])

        if play_indices.numel() > 0:
            play_rows = row_indices_long.index_select(0, play_indices)
            play_hand_indices = scoring_plan.arg0.index_select(0, play_indices)
            play_stage_slots = scoring_plan.arg1.index_select(0, play_indices)
            _play_target_context, play_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                play_rows,
                play_stage_slots,
            )
            del _play_target_context
            play_score = self._slot_preference_values(play_stage_slots, dtype=dtype)
            play_score = play_score + torch.where(
                play_stage_slots <= 2,
                play_stage_slots.new_full(play_stage_slots.shape, float(scoring_profile.play_front_bonus), dtype=dtype),
                torch.where(
                    play_stage_slots <= 4,
                    play_stage_slots.new_full(
                        play_stage_slots.shape, float(scoring_profile.play_back_bonus), dtype=dtype
                    ),
                    play_stage_slots.new_zeros(play_stage_slots.shape, dtype=dtype),
                ),
            )
            play_score = torch.where(
                play_target_numeric[:, 0].to(dtype=dtype) > 0.5,
                play_stage_slots.new_full(play_stage_slots.shape, -1000.0, dtype=dtype),
                play_score,
            )
            score0.index_fill_(0, play_indices, float(scoring_profile.play_priority))
            score1.index_copy_(0, play_indices, play_score)
            score2.index_copy_(0, play_indices, self._public_prefer_lower(play_hand_indices, dtype=dtype))

        if hand_indices.numel() > 0:
            hand_rows = row_indices_long.index_select(0, hand_indices)
            hand_family_ids = scoring_plan.family_ids.index_select(0, hand_indices)
            hand_indices_values = scoring_plan.arg0.index_select(0, hand_indices)
            if self._climax_play_family_id >= 0:
                climax_mask = hand_family_ids == self._climax_play_family_id
                if bool(climax_mask.any().item()):
                    climax_indices = hand_indices[climax_mask]
                    climax_rows = hand_rows[climax_mask]
                    score0.index_fill_(0, climax_indices, float(scoring_profile.climax_priority))
                    score1.index_copy_(
                        0,
                        climax_indices,
                        attackers_available.index_select(0, climax_rows) * float(scoring_profile.climax_attacker_scale)
                        + front_defenders.index_select(0, climax_rows) * float(scoring_profile.climax_defender_scale)
                        + torch.where(
                            attackers_available.index_select(0, climax_rows) > 0.0,
                            hand_indices_values.new_full(
                                climax_rows.shape,
                                float(scoring_profile.climax_active_bonus),
                                dtype=dtype,
                            ),
                            hand_indices_values.new_full(
                                climax_rows.shape,
                                float(scoring_profile.climax_inactive_bonus),
                                dtype=dtype,
                            ),
                        ),
                    )
                    score2.index_copy_(
                        0, climax_indices, self._public_prefer_lower(hand_indices_values[climax_mask], dtype=dtype)
                    )
            if self._clock_from_hand_family_id >= 0:
                clock_mask = hand_family_ids == self._clock_from_hand_family_id
                if bool(clock_mask.any().item()):
                    clock_indices = hand_indices[clock_mask]
                    clock_rows = hand_rows[clock_mask]
                    level_counts = self_level_count.index_select(0, clock_rows)
                    clock_counts = self_clock_count.index_select(0, clock_rows)
                    score0.index_fill_(0, clock_indices, float(scoring_profile.clock_priority))
                    score1.index_copy_(
                        0,
                        clock_indices,
                        torch.where(
                            (level_counts <= 0.0) & (clock_counts < 6.0),
                            float(scoring_profile.early_clock_score) - clock_counts,
                            clock_counts.new_full(
                                clock_counts.shape, float(scoring_profile.late_clock_score), dtype=dtype
                            ),
                        ),
                    )
                    score2.index_copy_(
                        0, clock_indices, self._public_prefer_lower(hand_indices_values[clock_mask], dtype=dtype)
                    )
            if self._main_event_family_id >= 0:
                event_mask = hand_family_ids == self._main_event_family_id
                if bool(event_mask.any().item()):
                    event_indices = hand_indices[event_mask]
                    score0.index_fill_(0, event_indices, float(scoring_profile.event_priority))
                    score1.index_fill_(0, event_indices, 10.0)
                    score2.index_copy_(
                        0, event_indices, self._public_prefer_lower(hand_indices_values[event_mask], dtype=dtype)
                    )
            if self._mulligan_select_family_id >= 0:
                mulligan_mask = hand_family_ids == self._mulligan_select_family_id
                if bool(mulligan_mask.any().item()):
                    mulligan_indices = hand_indices[mulligan_mask]
                    score0.index_fill_(0, mulligan_indices, float(scoring_profile.mulligan_select_priority))
                    score1.index_copy_(
                        0, mulligan_indices, self._public_prefer_lower(hand_indices_values[mulligan_mask], dtype=dtype)
                    )

        if index_family_indices.numel() > 0:
            index_rows = row_indices_long.index_select(0, index_family_indices)
            index_family_ids = scoring_plan.family_ids.index_select(0, index_family_indices)
            index_values = scoring_plan.arg0.index_select(0, index_family_indices)
            if self._choice_select_family_id >= 0:
                choice_mask = index_family_ids == self._choice_select_family_id
                if bool(choice_mask.any().item()):
                    choice_indices = index_family_indices[choice_mask]
                    score0.index_fill_(0, choice_indices, float(scoring_profile.choice_select_priority))
                    score1.index_copy_(
                        0, choice_indices, self._public_prefer_lower(index_values[choice_mask], dtype=dtype)
                    )
            if self._level_up_family_id >= 0:
                level_up_mask = index_family_ids == self._level_up_family_id
                if bool(level_up_mask.any().item()):
                    level_indices = index_family_indices[level_up_mask]
                    score0.index_fill_(0, level_indices, float(scoring_profile.level_up_priority))
                    score1.index_copy_(
                        0, level_indices, self._public_prefer_lower(index_values[level_up_mask], dtype=dtype)
                    )
            if self._trigger_order_family_id >= 0:
                trigger_mask = index_family_ids == self._trigger_order_family_id
                if bool(trigger_mask.any().item()):
                    trigger_indices = index_family_indices[trigger_mask]
                    score0.index_fill_(0, trigger_indices, float(scoring_profile.trigger_order_priority))
                    score1.index_copy_(
                        0, trigger_indices, self._public_prefer_lower(index_values[trigger_mask], dtype=dtype)
                    )
            if self._next_page_family_id >= 0:
                next_mask = index_family_ids == self._next_page_family_id
                if bool(next_mask.any().item()):
                    next_indices = index_family_indices[next_mask]
                    next_rows = index_rows[next_mask]
                    score0.index_fill_(0, next_indices, float(scoring_profile.pager_priority))
                    score1.index_copy_(
                        0,
                        next_indices,
                        torch.clamp(
                            choice_total.index_select(0, next_rows)
                            - (choice_page_start.index_select(0, next_rows) + 16.0),
                            min=0.0,
                        ),
                    )
            if self._prev_page_family_id >= 0:
                prev_mask = index_family_ids == self._prev_page_family_id
                if bool(prev_mask.any().item()):
                    prev_indices = index_family_indices[prev_mask]
                    prev_rows = index_rows[prev_mask]
                    score0.index_fill_(0, prev_indices, float(scoring_profile.pager_priority))
                    score1.index_copy_(
                        0, prev_indices, torch.clamp(choice_page_start.index_select(0, prev_rows), min=0.0)
                    )

        if move_indices.numel() > 0:
            move_rows = row_indices_long.index_select(0, move_indices)
            move_from_slots = scoring_plan.arg0.index_select(0, move_indices)
            move_to_slots = scoring_plan.arg1.index_select(0, move_indices)
            _move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_from_slots,
            )
            del _move_source_context
            _move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                self_stage_numeric,
                move_rows,
                move_to_slots,
            )
            del _move_target_context
            source_pref = self._slot_preference_values(move_from_slots, dtype=dtype)
            target_pref = self._slot_preference_values(move_to_slots, dtype=dtype)
            bonus = torch.zeros(move_to_slots.shape, dtype=dtype, device=row_indices_long.device)
            bonus = bonus + (
                ((move_from_slots >= 3) & (move_to_slots <= 2)).to(dtype=dtype)
                * float(scoring_profile.move_back_to_front_bonus)
            )
            bonus = bonus + (
                (
                    (move_to_slots == _PUBLIC_HEURISTIC_CENTER_SLOT)
                    & (move_from_slots != _PUBLIC_HEURISTIC_CENTER_SLOT)
                ).to(dtype=dtype)
                * float(scoring_profile.move_center_bonus)
            )
            legal = (move_source_numeric[:, 0].to(dtype=dtype) > 0.5) & (
                move_target_numeric[:, 0].to(dtype=dtype) <= 0.5
            )
            move_score = torch.where(
                legal,
                (target_pref - source_pref) + bonus,
                move_to_slots.new_full(move_to_slots.shape, -1000.0, dtype=dtype),
            )
            score0.index_fill_(0, move_indices, float(scoring_profile.move_priority))
            score1.index_copy_(0, move_indices, move_score)

        if default_indices.numel() > 0:
            default_family_ids = scoring_plan.family_ids.index_select(0, default_indices)
            if self._mulligan_confirm_family_id >= 0:
                mulligan_confirm_mask = default_family_ids == self._mulligan_confirm_family_id
                if bool(mulligan_confirm_mask.any().item()):
                    score0.index_fill_(
                        0,
                        default_indices[mulligan_confirm_mask],
                        float(scoring_profile.mulligan_confirm_priority),
                    )
            if self._pass_family_id >= 0:
                pass_mask = default_family_ids == self._pass_family_id
                if bool(pass_mask.any().item()):
                    score0.index_fill_(0, default_indices[pass_mask], float(scoring_profile.pass_priority))

        return self._combine_public_heuristic_scores(score0, score1, score2, dtype=dtype)

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

    def _resolve_candidate_components(
        self,
        candidate_ids: Tensor,
        candidate_meta: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if candidate_meta is None:
            return (
                self._family_ids.index_select(0, candidate_ids),
                self._hand_indices.index_select(0, candidate_ids),
                self._stage_slots.index_select(0, candidate_ids),
                self._from_slots.index_select(0, candidate_ids),
                self._to_slots.index_select(0, candidate_ids),
                self._attack_slots.index_select(0, candidate_ids),
                self._attack_types.index_select(0, candidate_ids),
                self._generic_indices.index_select(0, candidate_ids),
            )
        family_ids = candidate_meta[:, 0].to(dtype=torch.long)
        arg0 = candidate_meta[:, 1].to(dtype=torch.long)
        arg1 = candidate_meta[:, 2].to(dtype=torch.long)
        meta_unused = torch.full_like(arg0, self._meta_unused)
        arg0 = torch.where(arg0 == meta_unused, torch.full_like(arg0, -1), arg0)
        arg1 = torch.where(arg1 == meta_unused, torch.full_like(arg1, -1), arg1)

        hand_indices = torch.full_like(arg0, -1)
        hand_family_ids = (
            self._play_character_family_id,
            self._main_event_family_id,
            self._clock_from_hand_family_id,
            self._climax_play_family_id,
            self._mulligan_select_family_id,
        )
        for family_id in hand_family_ids:
            if family_id < 0:
                continue
            family_mask = family_ids == family_id
            hand_indices[family_mask] = arg0[family_mask]

        stage_slots = torch.full_like(arg0, -1)
        if self._play_character_family_id >= 0:
            play_mask = family_ids == self._play_character_family_id
            stage_slots[play_mask] = arg1[play_mask]

        from_slots = torch.full_like(arg0, -1)
        to_slots = torch.full_like(arg0, -1)
        if self._main_move_family_id >= 0:
            move_mask = family_ids == self._main_move_family_id
            from_slots[move_mask] = arg0[move_mask]
            to_slots[move_mask] = arg1[move_mask]

        attack_slots = torch.full_like(arg0, -1)
        attack_types = torch.full_like(arg0, -1)
        if self._attack_family_id >= 0:
            attack_mask = family_ids == self._attack_family_id
            attack_slots[attack_mask] = arg0[attack_mask]
            attack_types[attack_mask] = arg1[attack_mask]

        generic_indices = torch.full_like(arg0, -1)
        generic_family_ids = (
            self._choice_select_family_id,
            self._level_up_family_id,
            self._trigger_order_family_id,
        )
        for family_id in generic_family_ids:
            if family_id < 0:
                continue
            generic_mask = family_ids == family_id
            generic_indices[generic_mask] = arg0[generic_mask]

        return (
            family_ids,
            hand_indices,
            stage_slots,
            from_slots,
            to_slots,
            attack_slots,
            attack_types,
            generic_indices,
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
        valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
        if not torch.any(valid):
            return (
                slot_contexts.new_zeros((slot_indices.shape[0], slot_contexts.shape[-1])),
                slot_numeric.new_zeros((slot_indices.shape[0], slot_numeric.shape[-1])),
            )
        safe_rows = torch.where(valid, row_indices, torch.zeros_like(row_indices)).to(dtype=torch.long)
        safe_slots = torch.where(valid, slot_indices, torch.zeros_like(slot_indices)).to(dtype=torch.long)
        flat_indices = safe_rows * self._stage_slot_count + safe_slots
        gathered_context = slot_contexts.reshape(-1, slot_contexts.shape[-1]).index_select(0, flat_indices)
        gathered_numeric = slot_numeric.reshape(-1, slot_numeric.shape[-1]).index_select(0, flat_indices)
        return (
            gathered_context * valid.unsqueeze(1).to(dtype=slot_contexts.dtype),
            gathered_numeric * valid.unsqueeze(1).to(dtype=slot_numeric.dtype),
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
        if slot_contexts.ndim == 3:
            valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
            safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
            context_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_contexts.shape[-1])
            numeric_index = safe_indices.to(dtype=torch.long).view(-1, 1, 1).expand(-1, 1, slot_numeric.shape[-1])
            gathered_context = torch.gather(slot_contexts, 1, context_index).squeeze(1)
            gathered_numeric = torch.gather(slot_numeric, 1, numeric_index).squeeze(1)
            return (
                gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
                gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
            )
        valid = (slot_indices >= 0) & (slot_indices < self._stage_slot_count)
        safe_indices = torch.where(valid, slot_indices, torch.zeros_like(slot_indices))
        gathered_context = slot_contexts.index_select(0, safe_indices.to(dtype=torch.long))
        gathered_numeric = slot_numeric.index_select(0, safe_indices.to(dtype=torch.long))
        return (
            gathered_context * valid.unsqueeze(-1).to(dtype=slot_contexts.dtype),
            gathered_numeric * valid.unsqueeze(-1).to(dtype=slot_numeric.dtype),
        )

    def _extract_card_vector(self, obs_batch: Tensor, observation_slice: ObservationSlice | None) -> Tensor:
        if observation_slice is None:
            return torch.zeros((obs_batch.shape[0], 0), device=obs_batch.device, dtype=torch.long)
        return obs_batch[:, observation_slice.start : observation_slice.stop].to(dtype=torch.long)

    def _slot_component(self, stage_values: Tensor, offset: int) -> Tensor:
        if offset >= stage_values.shape[-1]:
            return torch.zeros(stage_values.shape[:2], device=stage_values.device, dtype=stage_values.dtype)
        return stage_values[..., offset]


class StructuredLegalPolicyValueModel(PolicyValueModel):
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
        self.policy_head = _StructuredLegalActionHead(
            latent_width=int(structured_config.gru_hidden_size),
            action_catalog=action_catalog,
            observation_contract=observation_contract,
            card_table=cached_runtime_card_table() if card_table is None else card_table,
            action_feature_width=action_feature_width,
            layer_norm=bool(structured_config.layer_norm),
            dropout_p=float(encoder_dropout),
            candidate_scoring_chunk_size=int(structured_config.candidate_scoring_chunk_size),
            cuda_learner_candidate_scoring_chunk_size=int(structured_config.cuda_learner_candidate_scoring_chunk_size),
            public_heuristic_logit_bias_scale=float(structured_config.public_heuristic_logit_bias_scale),
            public_heuristic_actor_logit_bias_scale=float(structured_config.public_heuristic_actor_logit_bias_scale),
            public_heuristic_logit_bias_families=tuple(structured_config.public_heuristic_logit_bias_families),
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

    def encode(self, obs: Tensor) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        if self._card_scalar_indices.numel() == 0:
            return self.encoder(obs_batch)
        prepared = obs_batch * self._encoder_input_keep_mask.to(device=obs_batch.device, dtype=obs_batch.dtype)
        return self.encoder(prepared)

    def forward(
        self,
        obs: Tensor,
        hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
        scoring_mode: str = "auto",
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(
            recurrent_output,
            obs=obs_batch,
            legal_actions=legal_actions,
            scoring_mode=scoring_mode,
        )
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_sequence_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        recurrent_flat, flat_obs_batch, seat_hidden, time_steps, batch_size = self._sequence_recurrent_outputs(
            obs,
            acting_seat,
            seat_hidden_state,
        )
        logits_flat = self.policy_head.score_legal_actions(
            recurrent_flat,
            obs=flat_obs_batch,
            legal_actions=legal_actions,
        )
        value_flat = self.value_head(recurrent_flat).squeeze(-1)
        return (
            logits_flat.reshape(time_steps, batch_size, logits_flat.shape[-1]),
            value_flat.reshape(time_steps, batch_size),
            seat_hidden,
        )

    def forward_sequence_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "learner",
    ) -> tuple[Tensor, Tensor, Tensor]:
        recurrent_flat, state_repr, observation_context, values, seat_hidden = self.forward_trunk_sequence_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
        )
        packed_logits = self.score_packed_legal_candidates(
            recurrent_flat,
            obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        return packed_logits, values, seat_hidden

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> StructuredLegalPolicyValueModel:
        compiled_packed = self._compiled_trunk_packed_core
        compiled_sequence = self._compiled_trunk_sequence_core
        if compiled_packed is None:
            compiled_packed = torch.compile(
                self._forward_trunk_packed_core,
                mode=mode,
            )
        if compiled_sequence is None:
            compiled_sequence = torch.compile(
                self._forward_trunk_sequence_core,
                mode=mode,
            )
        self._compiled_trunk_packed_core = compiled_packed
        self._compiled_trunk_sequence_core = compiled_sequence
        return self

    def advance_seat_hidden(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        _, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return next_seat_hidden

    def value_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, _next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        return self.value_head(recurrent_output).squeeze(-1)

    def score_packed_legal_candidates(
        self,
        recurrent_outputs: Tensor,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        *,
        state_repr: Tensor | None = None,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        recurrent_batch = recurrent_outputs
        if recurrent_batch.ndim != 2:
            raise ValueError("recurrent_outputs must be 2D (rows, hidden)")
        obs_batch = self._require_observation_batch(obs)
        if legal_actions.ids is None or legal_actions.offsets is None or legal_actions.meta is None:
            raise ValueError("score_packed_legal_candidates requires packed ids, offsets, and metadata")
        return self.policy_head.score_packed_candidates(
            recurrent_batch,
            obs=obs_batch,
            legal_actions=legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )

    def score_packed_public_heuristic_candidates(
        self,
        obs: Tensor,
        legal_actions: LegalActionBatch,
        *,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_profile: str = "base",
    ) -> Tensor:
        obs_batch = self._require_observation_batch(obs)
        return self.policy_head.score_packed_public_heuristic_candidates(
            obs=obs_batch,
            legal_actions=legal_actions,
            observation_context=observation_context,
            scoring_profile=scoring_profile,
        )

    def evaluate_factorized_sequence_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        actions: Tensor | None = None,
        same_family_reference_actions: Tensor | None = None,
        same_family_reference_families: Tensor | None = None,
    ) -> _FactorizedEvaluationResult:
        recurrent_flat, state_repr, observation_context, values, _seat_hidden = self.forward_trunk_sequence_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
        )
        head_result = self.policy_head.evaluate_factorized_packed(
            recurrent_flat,
            obs=obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
            legal_actions=legal_actions,
            actions=None if actions is None else actions.reshape(-1),
            same_family_reference_actions=(
                None if same_family_reference_actions is None else same_family_reference_actions.reshape(-1)
            ),
            same_family_reference_families=(
                None if same_family_reference_families is None else same_family_reference_families.reshape(-1)
            ),
            state_repr=state_repr,
            observation_context=observation_context,
        )
        return _FactorizedEvaluationResult(
            values=values,
            action_logp=None
            if head_result.action_logp is None
            else head_result.action_logp.reshape(obs.shape[0], obs.shape[1]),
            entropy=None if head_result.entropy is None else head_result.entropy.reshape(obs.shape[0], obs.shape[1]),
            family_log_probs=head_result.family_log_probs.reshape(
                obs.shape[0], obs.shape[1], head_result.family_log_probs.shape[-1]
            ),
            play_slot_log_probs=(
                None
                if head_result.play_slot_log_probs is None
                else head_result.play_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.play_slot_log_probs.shape[-1],
                )
            ),
            move_source_log_probs=(
                None
                if head_result.move_source_log_probs is None
                else head_result.move_source_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.move_source_log_probs.shape[-1],
                )
            ),
            move_slot_log_probs=(
                None
                if head_result.move_slot_log_probs is None
                else head_result.move_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.move_slot_log_probs.shape[-1],
                )
            ),
            attack_slot_log_probs=(
                None
                if head_result.attack_slot_log_probs is None
                else head_result.attack_slot_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.attack_slot_log_probs.shape[-1],
                )
            ),
            attack_type_log_probs=(
                None
                if head_result.attack_type_log_probs is None
                else head_result.attack_type_log_probs.reshape(
                    obs.shape[0],
                    obs.shape[1],
                    head_result.attack_type_log_probs.shape[-1],
                )
            ),
            top_action_ids=(
                None
                if head_result.top_action_ids is None
                else head_result.top_action_ids.reshape(obs.shape[0], obs.shape[1])
            ),
            same_family_action_logp=(
                None
                if head_result.same_family_action_logp is None
                else head_result.same_family_action_logp.reshape(obs.shape[0], obs.shape[1])
            ),
            same_family_top_action_ids=(
                None
                if head_result.same_family_top_action_ids is None
                else head_result.same_family_top_action_ids.reshape(obs.shape[0], obs.shape[1])
            ),
        )

    def forward_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "actor",
    ) -> tuple[Tensor, Tensor, Tensor]:
        recurrent_output, state_repr, observation_context, value, next_seat_hidden = (
            self.forward_trunk_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
            )
        )
        packed_logits = self.score_packed_legal_candidates(
            recurrent_output,
            self._require_observation_batch(obs),
            legal_actions,
            state_repr=state_repr,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        return packed_logits, value, next_seat_hidden

    def sample_factorized_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        recurrent_output, state_repr, observation_context, value, next_seat_hidden = (
            self.forward_trunk_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden_state,
            )
        )
        actions, behavior_logp = self.policy_head.sample_factorized_packed(
            recurrent_output,
            obs=self._require_observation_batch(obs),
            legal_actions=legal_actions,
            sample_seeds=sample_seeds,
            pass_action_id=pass_action_id,
            state_repr=state_repr,
            observation_context=observation_context,
        )
        return actions, behavior_logp, value, next_seat_hidden

    def sample_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        sample_seeds: Tensor,
        pass_action_id: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        packed_logits, value, next_seat_hidden = self.forward_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
            legal_actions=legal_actions,
            scoring_mode="actor",
        )
        if legal_actions.ids is None or legal_actions.offsets is None:
            raise ValueError("sample_packed_seat_aware requires packed ids and offsets")
        actions, behavior_logp = _sample_packed_action_scores(
            packed_logits,
            torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long),
            torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long),
            sample_seeds.to(device=packed_logits.device, dtype=torch.long),
            pass_action_id=int(pass_action_id),
        )
        return actions, behavior_logp, value, next_seat_hidden

    def forward_trunk_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        trunk_forward = self._compiled_trunk_packed_core
        if trunk_forward is not None:
            try:
                recurrent_output, obs_batch, value, next_seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_packed_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
        else:
            recurrent_output, obs_batch, value, next_seat_hidden = self._forward_trunk_packed_core(
                obs,
                acting_seat,
                seat_hidden_state,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(recurrent_output, obs=obs_batch)
        return recurrent_output, state_repr, observation_context, value, next_seat_hidden

    def forward_trunk_sequence_seat_aware(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
        time_steps = int(obs.shape[0])
        batch_size = int(obs.shape[1])
        trunk_forward = self._compiled_trunk_sequence_core
        if trunk_forward is not None:
            try:
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = trunk_forward(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
            except Exception as exc:
                self._compiled_trunk_sequence_core = None
                self._trunk_compile_last_error = repr(exc)
                recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                    obs,
                    acting_seat,
                    seat_hidden_state,
                )
        else:
            recurrent_flat, flat_obs_batch, value_flat, seat_hidden = self._forward_trunk_sequence_core(
                obs,
                acting_seat,
                seat_hidden_state,
            )
        state_repr, observation_context = self.policy_head._build_state_representation(
            recurrent_flat,
            obs=flat_obs_batch,
        )
        return recurrent_flat, state_repr, observation_context, value_flat.reshape(time_steps, batch_size), seat_hidden

    def _forward_trunk_packed_core(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        value = self.value_head(recurrent_output).squeeze(-1)
        return recurrent_output, obs_batch, value, next_seat_hidden

    def _forward_trunk_sequence_core(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        recurrent_flat, flat_obs_batch, seat_hidden, _time_steps, _batch_size = self._sequence_recurrent_outputs(
            obs,
            acting_seat,
            seat_hidden_state,
        )
        value_flat = self.value_head(recurrent_flat).squeeze(-1)
        return recurrent_flat, flat_obs_batch, value_flat, seat_hidden

    def _sequence_recurrent_outputs(
        self,
        obs: Tensor,
        acting_seat: Tensor,
        seat_hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        time_steps, batch_size, obs_dim = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])
        flat_obs = obs.reshape(time_steps * batch_size, obs_dim)
        encoded_flat = self.encode(flat_obs)
        encoded = encoded_flat.reshape(time_steps, batch_size, encoded_flat.shape[-1])
        seat_hidden = self._prepare_seat_hidden_state(
            seat_hidden_state,
            batch_size=batch_size,
            like=encoded[0],
        )
        recurrent_steps: list[Tensor] = []
        for step_encoded, step_seat in zip(encoded.unbind(dim=0), acting_seat.unbind(dim=0), strict=True):
            recurrent_output, seat_hidden = self.recurrent_step_seat_aware(
                step_encoded,
                step_seat,
                seat_hidden,
            )
            recurrent_steps.append(recurrent_output)
        recurrent = torch.stack(recurrent_steps, dim=0)
        recurrent_flat = recurrent.reshape(time_steps * batch_size, recurrent.shape[-1])
        return recurrent_flat, self._require_observation_batch(flat_obs), seat_hidden, time_steps, batch_size


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
    "build_policy_value_model",
]
