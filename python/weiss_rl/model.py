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
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.observation_layout import ObservationLayout, ObservationPlayerBlock, ObservationSlice, parse_observation_layout

GLOBAL_ACTION_SPACE_SIZE = 527
SEAT_COUNT = 2
STRUCTURED_V2_ENCODER_KIND = "structured_v2"


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
    stage_slot_count: int
    sentinel_hidden: int
    sentinel_empty_card: int
    card_scalar_indices: tuple[int, ...]


def _slice_by_name(block: ObservationPlayerBlock, name: str) -> ObservationSlice | None:
    for current in block.slices:
        if current.name == name:
            return current
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
        fallback_positions = packed_offsets[1:].to(device=packed_scores.device, dtype=torch.long).index_select(
            0, non_empty_rows.index_select(0, missing_rows)
        ) - 1
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

    def forward(self, obs: Tensor, hidden_state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
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
            raise ValueError(
                "acting_seat must be 2D (time, batch) with the same leading dimensions as obs"
            )
        time_steps, batch_size = int(obs.shape[0]), int(obs.shape[1])
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
    ) -> None:
        super().__init__()
        if latent_width <= 0:
            raise ValueError(f"latent_width must be >= 1, got {latent_width}")
        if action_feature_width <= 0:
            raise ValueError(f"action_feature_width must be >= 1, got {action_feature_width}")
        self.action_dim = int(action_catalog.action_space_size)
        self._stage_slot_count = max(int(action_catalog.max_stage), 1)
        self._observation_contract = observation_contract
        self._card_vocab_size = 32768

        family_names = tuple(family.name for family in action_catalog.families)
        family_index = {name: index for index, name in enumerate(family_names)}
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
        self._main_move_family_id = int(family_index.get("main_move", -1))
        self._choice_select_family_id = int(family_index.get("choice_select", -1))
        self._level_up_family_id = int(family_index.get("level_up", -1))
        self._trigger_order_family_id = int(family_index.get("trigger_order", -1))
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
        self._candidate_scoring_chunk_size = 65536
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
            self._encode_observation_context(obs_batch)
            if observation_context is None
            else dict(observation_context)
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
        meta = None if legal_actions.meta is None else torch.as_tensor(legal_actions.meta, device=latent.device, dtype=torch.long)
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
        )

    def forward(
        self,
        latent: Tensor,
        *,
        obs: Tensor,
        legal_actions: LegalActionBatch | None = None,
    ) -> Tensor:
        return self.score_legal_actions(latent, obs=obs, legal_actions=legal_actions)

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
            "opponent_stage_context": opponent_stage_ctx,
            "opponent_stage_numeric": opponent_stage_numeric,
            "opponent_stage_summary": opponent_stage_ctx.mean(dim=1),
        }

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
        stage_values = obs_batch[:, stage_slice.start : stage_slice.stop].reshape(batch_size, self._stage_slot_count, slot_width)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = family_ids.device
        play_mask = family_ids == self._play_character_family_id
        hand_mask = torch.zeros_like(play_mask)
        for family_id in self._hand_family_ids:
            hand_mask |= family_ids == family_id
        move_mask = family_ids == self._main_move_family_id
        attack_mask = family_ids == self._attack_family_id
        default_mask = ~(play_mask | hand_mask | move_mask | attack_mask)

        def _indices(mask: Tensor) -> Tensor:
            if not torch.any(mask):
                return torch.zeros((0,), device=device, dtype=torch.long)
            return torch.nonzero(mask, as_tuple=False).squeeze(1)

        return (
            _indices(play_mask),
            _indices(hand_mask),
            _indices(move_mask),
            _indices(attack_mask),
            _indices(default_mask),
        )

    def _score_candidates_chunked(
        self,
        state_repr: Tensor,
        row_indices: Tensor,
        candidate_ids: Tensor,
        observation_context: Mapping[str, Tensor],
        *,
        candidate_meta: Tensor | None = None,
    ) -> Tensor:
        if candidate_ids.numel() == 0:
            return state_repr.new_zeros((0,))
        scores_chunks: list[Tensor] = []
        chunk_size = max(1, int(self._candidate_scoring_chunk_size))
        for start in range(0, int(candidate_ids.numel()), chunk_size):
            end = min(start + chunk_size, int(candidate_ids.numel()))
            scores_chunks.append(
                self._score_candidates(
                    state_repr,
                    row_indices[start:end],
                    candidate_ids[start:end],
                    observation_context,
                    candidate_meta=None if candidate_meta is None else candidate_meta[start:end],
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
        for start in range(0, scoring_plan.candidate_count, chunk_size):
            end = min(start + chunk_size, scoring_plan.candidate_count)
            scores_chunks.append(
                self._score_packed_candidates_plan(
                    state_repr,
                    scoring_plan.slice(start, end),
                    observation_context,
                    scoring_mode=scoring_mode,
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
        play_indices, hand_indices, move_indices, attack_indices, default_indices = self._partition_candidate_family_indices(
            scoring_plan.family_ids
        )

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
                observation_context["self_stage_numeric"],
                play_rows,
                play_stage_slots,
            )
            scores.index_copy_(
                0,
                play_indices,
                self._score_candidate_group(
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
                ),
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
            scores.index_copy_(
                0,
                hand_indices,
                self._score_candidate_group(
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
                ),
            )

        if move_indices.numel() > 0:
            move_rows = row_indices_long.index_select(0, move_indices)
            move_row_states = row_states.index_select(0, move_indices)
            move_from_slots = scoring_plan.arg0.index_select(0, move_indices)
            move_to_slots = scoring_plan.arg1.index_select(0, move_indices)
            move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                move_rows,
                move_from_slots,
            )
            move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                move_rows,
                move_to_slots,
            )
            scores.index_copy_(
                0,
                move_indices,
                self._score_candidate_group(
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
                ),
            )

        if attack_indices.numel() > 0:
            attack_rows = row_indices_long.index_select(0, attack_indices)
            attack_row_states = row_states.index_select(0, attack_indices)
            attack_slot_values = scoring_plan.arg0.index_select(0, attack_indices)
            attack_type_values = scoring_plan.arg1.index_select(0, attack_indices)
            attack_source_context, _attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                attack_rows,
                attack_slot_values,
            )
            defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                observation_context["opponent_stage_numeric"],
                attack_rows,
                attack_slot_values,
            )
            scores.index_copy_(
                0,
                attack_indices,
                self._score_candidate_group(
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
                ),
            )

        if default_indices.numel() > 0:
            default_row_states = row_states.index_select(0, default_indices)
            default_generic_indices = scoring_plan.arg0.index_select(0, default_indices)
            scores.index_copy_(
                0,
                default_indices,
                self._score_candidate_group(
                    default_row_states,
                    feature_sections=(
                        (
                            family_embeddings.index_select(0, default_indices),
                            (self._family_feature_offset, self._hand_card_feature_offset),
                        ),
                    ),
                    numeric_sections=(
                        ((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),
                    ),
                    constant_numeric_ones=(8, 9),
                    scoring_mode=scoring_mode,
                ),
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
                projected = projected + linear.weight.index_select(1, constant_columns).sum(dim=1).to(dtype=projected.dtype)
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
            return self.joint_scorer(torch.cat([row_states, candidate_repr], dim=1)).squeeze(-1).to(dtype=row_states.dtype)
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
                observation_context["self_stage_numeric"],
                play_rows,
                play_stage_slots,
            )
            scores[play_mask] = self._score_candidate_group(
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

        move_mask = family_ids == self._main_move_family_id
        if torch.any(move_mask):
            move_rows = row_indices_long[move_mask]
            move_row_states = row_states[move_mask]
            move_from_slots = meta_arg0[move_mask] if meta_arg0 is not None else from_slots[move_mask]
            move_to_slots = meta_arg1[move_mask] if meta_arg1 is not None else to_slots[move_mask]
            move_source_context, move_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                move_rows,
                move_from_slots,
            )
            move_target_context, move_target_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                move_rows,
                move_to_slots,
            )
            scores[move_mask] = self._score_candidate_group(
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

        attack_mask = family_ids == self._attack_family_id
        if torch.any(attack_mask):
            attack_rows = row_indices_long[attack_mask]
            attack_row_states = row_states[attack_mask]
            attack_slot_values = meta_arg0[attack_mask] if meta_arg0 is not None else attack_slots[attack_mask]
            attack_type_values = meta_arg1[attack_mask] if meta_arg1 is not None else attack_types[attack_mask]
            attack_source_context, _attack_source_numeric = self._gather_stage_features_for_rows(
                observation_context["self_stage_context"],
                observation_context["self_stage_numeric"],
                attack_rows,
                attack_slot_values,
            )
            defender_context, defender_numeric = self._gather_stage_features_for_rows(
                observation_context["opponent_stage_context"],
                observation_context["opponent_stage_numeric"],
                attack_rows,
                attack_slot_values,
            )
            scores[attack_mask] = self._score_candidate_group(
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

        default_mask = ~(play_mask | hand_mask | move_mask | attack_mask)
        if torch.any(default_mask):
            default_row_states = row_states[default_mask]
            default_generic_indices = meta_arg0[default_mask] if meta_arg0 is not None else generic_indices[default_mask]
            scores[default_mask] = self._score_candidate_group(
                default_row_states,
                feature_sections=((family_embeddings[default_mask], (self._family_feature_offset, self._hand_card_feature_offset)),),
                numeric_sections=(
                    ((default_generic_indices >= 0).to(dtype=row_states.dtype).unsqueeze(1), (6,)),
                ),
                constant_numeric_ones=(8, 9),
            )

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
        static_features = self._card_static_features.index_select(0, bucketed_ids.reshape(-1)).reshape(
            *bucketed_ids.shape,
            self._card_static_features.shape[1],
        )
        projected = self.card_feature_projection(static_features.to(dtype=dtype))
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        logits = self.policy_head(recurrent_output, obs=obs_batch, legal_actions=legal_actions)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output, obs=obs_batch, legal_actions=legal_actions)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs_batch = self._require_observation_batch(obs)
        encoded_obs = self.encode(obs_batch)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        logits = self.policy_head(recurrent_output, obs=obs_batch, legal_actions=legal_actions)
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
            raise ValueError(
                "acting_seat must be 2D (time, batch) with the same leading dimensions as obs"
            )
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

    def forward_packed_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        legal_actions: LegalActionBatch,
        scoring_mode: str = "actor",
    ) -> tuple[Tensor, Tensor, Tensor]:
        recurrent_output, state_repr, observation_context, value, next_seat_hidden = self.forward_trunk_packed_seat_aware(
            obs,
            acting_seat,
            seat_hidden_state,
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
            raise ValueError(
                "acting_seat must be 2D (time, batch) with the same leading dimensions as obs"
            )
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
