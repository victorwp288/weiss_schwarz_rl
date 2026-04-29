"""Torch recurrent actor-critic model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

import weiss_rl.structured_action_head as _structured_action_head
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.card_table import cached_runtime_card_table
from weiss_rl.config.models import ModelConfig
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.model_layers import build_mlp_stack as _build_mlp_stack
from weiss_rl.observation_layout import (
    ObservationLayout,
    ObservationPlayerBlock,
    ObservationSlice,
    parse_observation_layout,
)
from weiss_rl.structured_action_head import (
    _build_structured_observation_contract,
    _FactorizedEvaluationResult,
    _StructuredLegalActionHead,
    _uniform_from_seeds,
)

GLOBAL_ACTION_SPACE_SIZE = 527
SEAT_COUNT = 2
STRUCTURED_V2_ENCODER_KIND = "structured_v2"


def _negative_logits_fill_value(dtype: torch.dtype) -> float:
    return _structured_action_head._negative_logits_fill_value(dtype)


def _packed_local_cdf(probabilities: Tensor, offsets: Tensor) -> Tensor:
    return _structured_action_head._packed_local_cdf(probabilities, offsets)


def _sample_packed_action_scores(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
    original_uniform_from_seeds = _structured_action_head._uniform_from_seeds
    original_packed_local_cdf = _structured_action_head._packed_local_cdf
    _structured_action_head._uniform_from_seeds = _uniform_from_seeds
    _structured_action_head._packed_local_cdf = _packed_local_cdf
    try:
        return _structured_action_head._sample_packed_action_scores(*args, **kwargs)
    finally:
        _structured_action_head._uniform_from_seeds = original_uniform_from_seeds
        _structured_action_head._packed_local_cdf = original_packed_local_cdf


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

    def set_public_heuristic_logit_bias_scale(
        self,
        value: float,
        *,
        actor_value: float | None = None,
    ) -> None:
        del value, actor_value

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
        del scoring_mode
        return 0.0

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
            public_heuristic_logit_bias_profile=str(structured_config.public_heuristic_logit_bias_profile),
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

    def set_public_heuristic_logit_bias_scale(
        self,
        value: float,
        *,
        actor_value: float | None = None,
    ) -> None:
        self.policy_head.set_public_heuristic_logit_bias_scales(
            learner_scale=float(value),
            actor_scale=None if actor_value is None else float(actor_value),
        )

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str = "learner") -> float:
        return float(self.policy_head._public_heuristic_logit_bias_scale_for(scoring_mode))

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
    if (
        float(config.public_heuristic_logit_bias_scale) != 0.0
        or float(config.public_heuristic_actor_logit_bias_scale) != -1.0
        or int(config.public_heuristic_logit_bias_start_updates) != 0
        or int(config.public_heuristic_logit_bias_end_updates) != -1
        or float(config.public_heuristic_logit_bias_final_scale) != 0.0
        or bool(config.public_heuristic_logit_bias_families)
        or str(config.public_heuristic_logit_bias_profile).strip().lower() != "base"
    ):
        raise ValueError(
            "public_heuristic_* model settings require encoder_kind='structured_v2'; "
            f"got encoder_kind={config.encoder_kind!r}"
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
