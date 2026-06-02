"""Base recurrent policy/value model methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any, cast

import torch
from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models import typed_encoder as model_typed_encoder
from weiss_rl.models.state import (
    prepare_acting_seat,
    prepare_hidden_state,
    prepare_seat_hidden_state,
    require_observation_batch,
    select_acting_hidden,
    write_acting_hidden,
)

SEAT_COUNT = 2
STRUCTURED_V2_ENCODER_KIND = "structured_v2"


class PolicyValueModelBaseMixin:
    observation_dim: int
    hidden_size: int
    recurrent_core: str
    encoder: nn.Module
    gru: nn.GRU | None
    feedforward_core: nn.Module | None
    policy_head: nn.Module
    value_head: nn.Module

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
        opponent_policy_ids: Sequence[object] | None = None,
        opponent_context_indices: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        hidden_device, hidden_dtype = self._hidden_tensor_device_dtype(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        hidden = torch.zeros(batch_size, SEAT_COUNT, self.hidden_size, device=hidden_device, dtype=hidden_dtype)
        context = self._opponent_context_hidden(
            batch_size=batch_size,
            device=hidden_device,
            dtype=hidden_dtype,
            opponent_policy_ids=opponent_policy_ids,
            opponent_context_indices=opponent_context_indices,
        )
        if context is not None:
            hidden = hidden + context.unsqueeze(1)
        return hidden

    def opponent_context_indices_for_policy_ids(
        self,
        opponent_policy_ids: Sequence[object],
        *,
        batch_size: int | None = None,
    ) -> list[int]:
        policy_ids = list(opponent_policy_ids)
        if batch_size is not None and len(policy_ids) != int(batch_size):
            raise ValueError(f"opponent_policy_ids must have length {int(batch_size)}, got {len(policy_ids)}")
        index_by_policy_id = getattr(self, "_opponent_context_index_by_policy_id", {})
        if not isinstance(index_by_policy_id, Mapping) or not index_by_policy_id:
            return [0 for _ in policy_ids]
        result: list[int] = []
        for policy_id in policy_ids:
            policy_text = str(policy_id).strip()
            exact = index_by_policy_id.get(policy_text)
            if exact is not None:
                result.append(int(exact))
                continue
            suffix_match = 0
            for configured_policy_id, configured_index in index_by_policy_id.items():
                if policy_text.endswith(f"_{configured_policy_id}"):
                    suffix_match = int(configured_index)
                    break
            result.append(suffix_match)
        return result

    def should_apply_opponent_context_for_eval_policy(self, policy_id: str) -> bool:
        enabled = cast(Set[str], getattr(self, "opponent_context_eval_policy_ids", frozenset()))
        return str(policy_id).strip() in enabled

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
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_hidden = self.recurrent_step(encoded_obs, hidden_state)
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_hidden

    def forward_seat_aware(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
        value = self.value_head(recurrent_output).squeeze(-1)
        return logits, value, next_seat_hidden

    def forward_seat_aware_inplace(
        self,
        obs: Tensor,
        acting_seat: int | Tensor,
        seat_hidden_state: Tensor | None = None,
        *,
        scoring_mode: str = "auto",
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del scoring_mode
        encoded_obs = self.encode(obs)
        recurrent_output, next_seat_hidden = self.recurrent_step_seat_aware_inplace(
            encoded_obs,
            acting_seat,
            seat_hidden_state,
        )
        recurrent_output = self._apply_opponent_context_recurrent_adapter(recurrent_output, opponent_context_index)
        logits = self.policy_head(recurrent_output)
        logits = self._apply_opponent_context_action_bias(logits, opponent_context_index)
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
        reset_before_step: Tensor | None = None,
        opponent_context_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if legal_actions is not None:
            raise ValueError("forward_sequence_seat_aware with legal_actions is only supported on structured models")
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")
        if acting_seat.ndim != 2 or acting_seat.shape != obs.shape[:2]:
            raise ValueError("acting_seat must be 2D (time, batch) with the same leading dimensions as obs")
        batch_size = int(obs.shape[1])
        seat_hidden = self._prepare_seat_hidden_state(seat_hidden_state, batch_size=batch_size, like=obs[0])
        reset_mask = None
        if reset_before_step is not None:
            reset_mask = torch.as_tensor(reset_before_step, device=obs.device, dtype=torch.bool)
            if reset_mask.ndim != 2 or reset_mask.shape != obs.shape[:2]:
                raise ValueError("reset_before_step must be 2D (time, batch) with the same leading dimensions as obs")
        context_index = None
        if opponent_context_index is not None:
            context_index = torch.as_tensor(opponent_context_index, device=obs.device, dtype=torch.long)
            if context_index.ndim != 2 or context_index.shape != obs.shape[:2]:
                raise ValueError(
                    "opponent_context_index must be 2D (time, batch) with the same leading dimensions as obs"
                )
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []
        for step_index, (step_obs, step_seat) in enumerate(
            zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)
        ):
            if reset_mask is not None:
                step_reset = reset_mask[step_index]
                if bool(step_reset.any().item()):
                    reset_rows = torch.nonzero(step_reset, as_tuple=False).squeeze(1)
                    seat_hidden = seat_hidden.clone()
                    seat_hidden.index_copy_(
                        0,
                        reset_rows,
                        self.initial_seat_hidden(
                            int(reset_rows.numel()),
                            device=seat_hidden.device,
                            dtype=seat_hidden.dtype,
                            opponent_context_indices=(
                                None if context_index is None else context_index[step_index].index_select(0, reset_rows)
                            ),
                        ),
                    )
            step_logits, step_value, seat_hidden = self.forward_seat_aware(
                step_obs,
                step_seat,
                seat_hidden,
                opponent_context_index=(None if context_index is None else context_index[step_index]),
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
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del temperature
        raise ValueError("sample_packed_seat_aware is only supported on structured models")

    def enable_trunk_compile(self, *, mode: str = "reduce-overhead") -> Any:
        del mode
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
        return model_typed_encoder.build_observation_encoder(
            observation_dim=observation_dim,
            config=config,
            observation_spec=observation_spec,
            dropout_p=dropout_p,
            structured_encoder_kind=STRUCTURED_V2_ENCODER_KIND,
        )

    def _require_observation_batch(self, obs: Tensor) -> Tensor:
        return require_observation_batch(
            obs,
            observation_dim=self.observation_dim,
            dtype=self._reference_parameter().dtype,
        )

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

    def _opponent_context_hidden(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        opponent_policy_ids: Sequence[object] | None,
        opponent_context_indices: Sequence[int] | Tensor | None,
    ) -> Tensor | None:
        offsets = getattr(self, "_opponent_context_hidden_offsets", None)
        adapter = getattr(self, "opponent_context_hidden_adapter", None)
        offset_rows = 0 if offsets is None else int(getattr(offsets, "shape", (0,))[0])
        adapter_rows = 0 if adapter is None else int(getattr(adapter, "shape", (0,))[0])
        row_count = max(offset_rows, adapter_rows)
        if row_count <= 1:
            return None
        if opponent_context_indices is None:
            if opponent_policy_ids is None:
                return None
            opponent_context_indices = self.opponent_context_indices_for_policy_ids(
                opponent_policy_ids,
                batch_size=batch_size,
            )
        indices = torch.as_tensor(opponent_context_indices, device=device, dtype=torch.long).reshape(-1)
        if int(indices.numel()) != int(batch_size):
            raise ValueError(f"opponent_context_indices must have length {int(batch_size)}, got {int(indices.numel())}")
        has_nonzero_context = bool((indices != 0).any().item())
        if not has_nonzero_context:
            return None
        indices = indices.clamp(min=0, max=row_count - 1)
        context = torch.zeros((int(batch_size), self.hidden_size), device=device, dtype=dtype)
        if offsets is not None and offset_rows > 1:
            offset_indices = indices.clamp(min=0, max=offset_rows - 1)
            context = context + offsets.to(device=device, dtype=dtype).index_select(0, offset_indices)
        if adapter is not None and adapter_rows > 1:
            adapter_scale = float(getattr(self, "opponent_context_trainable_hidden_scale", 1.0))
            if adapter_scale != 0.0:
                adapter_indices = indices.clamp(min=0, max=adapter_rows - 1)
                context = (
                    context + adapter.to(device=device, dtype=dtype).index_select(0, adapter_indices) * adapter_scale
                )
        context = context.masked_fill((indices == 0).unsqueeze(1), 0.0)
        return context

    def _opponent_context_indices_tensor(
        self,
        opponent_context_index: Tensor | None,
        *,
        row_count: int,
        device: torch.device,
        adapter_name: str = "opponent_context_action_bias_adapter",
    ) -> Tensor | None:
        adapter = getattr(self, adapter_name, None)
        if adapter is None or int(getattr(adapter, "shape", (0,))[0]) <= 1:
            return None
        if opponent_context_index is None:
            return None
        indices = torch.as_tensor(opponent_context_index, device=device, dtype=torch.long).reshape(-1)
        if int(indices.numel()) != int(row_count):
            raise ValueError(f"opponent_context_index must have length {int(row_count)}, got {int(indices.numel())}")
        if not bool((indices != 0).any().item()):
            return None
        return indices.clamp(min=0, max=int(adapter.shape[0]) - 1)

    def _has_opponent_context_action_bias(
        self,
        opponent_context_index: Tensor | None,
        *,
        row_count: int,
        device: torch.device,
    ) -> bool:
        scale = float(getattr(self, "opponent_context_trainable_action_bias_scale", 0.0))
        if scale == 0.0:
            return False
        return (
            self._opponent_context_indices_tensor(
                opponent_context_index,
                row_count=int(row_count),
                device=device,
                adapter_name="opponent_context_action_bias_adapter",
            )
            is not None
        )

    def _has_opponent_context_candidate_residual(
        self,
        opponent_context_index: Tensor | None,
        *,
        row_count: int,
        device: torch.device,
    ) -> bool:
        scale = float(getattr(self, "opponent_context_trainable_candidate_residual_scale", 0.0))
        if scale == 0.0:
            return False
        return (
            self._opponent_context_indices_tensor(
                opponent_context_index,
                row_count=int(row_count),
                device=device,
                adapter_name="opponent_context_candidate_residual_context",
            )
            is not None
        )

    def _has_opponent_context_packed_adjustment(
        self,
        opponent_context_index: Tensor | None,
        *,
        row_count: int,
        device: torch.device,
    ) -> bool:
        return self._has_opponent_context_action_bias(
            opponent_context_index,
            row_count=row_count,
            device=device,
        ) or self._has_opponent_context_candidate_residual(
            opponent_context_index,
            row_count=row_count,
            device=device,
        )

    def _apply_opponent_context_recurrent_adapter(
        self,
        recurrent_outputs: Tensor,
        opponent_context_index: Tensor | None,
    ) -> Tensor:
        adapter = getattr(self, "opponent_context_recurrent_adapter", None)
        scale = float(getattr(self, "opponent_context_trainable_recurrent_scale", 0.0))
        if adapter is None or scale == 0.0:
            return recurrent_outputs
        if recurrent_outputs.ndim != 2:
            raise ValueError(
                f"recurrent_outputs must be 2D (batch, hidden), got shape {tuple(recurrent_outputs.shape)}"
            )
        indices = self._opponent_context_indices_tensor(
            opponent_context_index,
            row_count=int(recurrent_outputs.shape[0]),
            device=recurrent_outputs.device,
            adapter_name="opponent_context_recurrent_adapter",
        )
        if indices is None:
            return recurrent_outputs
        bias = adapter.to(device=recurrent_outputs.device, dtype=recurrent_outputs.dtype).index_select(0, indices)
        bias = bias.masked_fill((indices == 0).unsqueeze(1), 0.0)
        return recurrent_outputs + bias * scale

    def _apply_opponent_context_action_bias(
        self,
        logits: Tensor,
        opponent_context_index: Tensor | None,
    ) -> Tensor:
        adapter = getattr(self, "opponent_context_action_bias_adapter", None)
        scale = float(getattr(self, "opponent_context_trainable_action_bias_scale", 0.0))
        if adapter is None or scale == 0.0:
            return logits
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D (batch, action), got shape {tuple(logits.shape)}")
        indices = self._opponent_context_indices_tensor(
            opponent_context_index,
            row_count=int(logits.shape[0]),
            device=logits.device,
            adapter_name="opponent_context_action_bias_adapter",
        )
        if indices is None:
            return logits
        bias = adapter.to(device=logits.device, dtype=logits.dtype).index_select(0, indices) * scale
        return logits + bias.masked_fill((indices == 0).unsqueeze(1), 0.0)

    def _apply_opponent_context_packed_action_bias(
        self,
        packed_logits: Tensor,
        legal_actions: LegalActionBatch,
        opponent_context_index: Tensor | None,
    ) -> Tensor:
        adapter = getattr(self, "opponent_context_action_bias_adapter", None)
        scale = float(getattr(self, "opponent_context_trainable_action_bias_scale", 0.0))
        if adapter is None or scale == 0.0:
            return packed_logits
        if legal_actions.ids is None or legal_actions.offsets is None:
            return packed_logits
        if packed_logits.ndim != 1:
            raise ValueError(f"packed_logits must be 1D, got shape {tuple(packed_logits.shape)}")
        offsets = torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long)
        row_count = int(offsets.numel() - 1)
        indices = self._opponent_context_indices_tensor(
            opponent_context_index,
            row_count=row_count,
            device=packed_logits.device,
            adapter_name="opponent_context_action_bias_adapter",
        )
        if indices is None:
            return packed_logits
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.sum().item()) != int(packed_logits.shape[0]):
            raise ValueError("packed legal offsets must align with packed logits")
        row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_logits.device), lengths)
        action_ids = torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long)
        if int(action_ids.numel()) != int(packed_logits.shape[0]):
            raise ValueError("packed legal ids must align with packed logits")
        row_context = indices.index_select(0, row_indices)
        bias_table = adapter.to(device=packed_logits.device, dtype=packed_logits.dtype)
        bias = bias_table[row_context, action_ids.clamp(min=0, max=int(bias_table.shape[1]) - 1)] * scale
        bias = bias.masked_fill(row_context == 0, 0.0)
        return packed_logits + bias

    def _apply_opponent_context_packed_candidate_residual(
        self,
        packed_logits: Tensor,
        legal_actions: LegalActionBatch,
        state_repr: Tensor,
        opponent_context_index: Tensor | None,
        *,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        context_table = getattr(self, "opponent_context_candidate_residual_context", None)
        state_layer = getattr(self, "opponent_context_candidate_residual_state", None)
        candidate_layer = getattr(self, "opponent_context_candidate_residual_candidate", None)
        meta_layer = getattr(self, "opponent_context_candidate_residual_meta", None)
        out_layer = getattr(self, "opponent_context_candidate_residual_out", None)
        scale = float(getattr(self, "opponent_context_trainable_candidate_residual_scale", 0.0))
        if context_table is None or state_layer is None or meta_layer is None or out_layer is None or scale == 0.0:
            return packed_logits
        if legal_actions.offsets is None or legal_actions.meta is None:
            return packed_logits
        if packed_logits.ndim != 1:
            raise ValueError(f"packed_logits must be 1D, got shape {tuple(packed_logits.shape)}")
        if state_repr.ndim != 2:
            raise ValueError(f"state_repr must be 2D, got shape {tuple(state_repr.shape)}")
        offsets = torch.as_tensor(legal_actions.offsets, device=packed_logits.device, dtype=torch.long)
        row_count = int(offsets.numel() - 1)
        if int(state_repr.shape[0]) != row_count:
            raise ValueError(f"state_repr row count must match packed offsets, got {int(state_repr.shape[0])}")
        indices = self._opponent_context_indices_tensor(
            opponent_context_index,
            row_count=row_count,
            device=packed_logits.device,
            adapter_name="opponent_context_candidate_residual_context",
        )
        if indices is None:
            return packed_logits
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.sum().item()) != int(packed_logits.shape[0]):
            raise ValueError("packed legal offsets must align with packed logits")
        if int(packed_logits.shape[0]) == 0:
            return packed_logits
        row_indices = torch.repeat_interleave(torch.arange(row_count, device=packed_logits.device), lengths)
        row_context = indices.index_select(0, row_indices)
        raw_meta = torch.as_tensor(legal_actions.meta, device=packed_logits.device, dtype=torch.float32)
        if raw_meta.ndim != 2 or raw_meta.shape[0] != packed_logits.shape[0] or raw_meta.shape[1] < 3:
            raise ValueError("packed legal meta must have shape (packed_actions, >=3)")
        meta = raw_meta[:, :3]
        meta = torch.where(meta >= 60000.0, torch.full_like(meta, -1.0), meta)
        meta_scale = meta.new_tensor([32.0, 64.0, 64.0])
        meta_features = meta / meta_scale
        row_state = state_repr.to(device=packed_logits.device, dtype=packed_logits.dtype).index_select(0, row_indices)
        context_features = context_table.to(device=packed_logits.device, dtype=packed_logits.dtype).index_select(
            0,
            row_context,
        )
        mode = str(getattr(self, "opponent_context_candidate_residual_mode", "additive")).strip().lower()
        if mode in {"rich", "rich_bilinear"}:
            if candidate_layer is None:
                raise ValueError("rich opponent-context candidate residual requires candidate residual layer")
            if observation_context is None:
                raise ValueError("rich opponent-context candidate residual requires observation_context")
            candidate_repr_fn = getattr(self.policy_head, "_project_packed_candidate_representations", None)
            if not callable(candidate_repr_fn):
                raise ValueError("rich opponent-context candidate residual requires packed candidate representations")
            candidate_repr = candidate_repr_fn(
                state_repr.to(device=packed_logits.device, dtype=packed_logits.dtype),
                legal_actions,
                observation_context,
                scoring_mode=scoring_mode,
            ).to(device=packed_logits.device, dtype=packed_logits.dtype)
            if tuple(candidate_repr.shape[:1]) != tuple(packed_logits.shape[:1]):
                raise ValueError("rich candidate residual representation must align with packed logits")
            candidate_features = (
                state_layer(row_state)
                + candidate_layer(candidate_repr)
                + meta_layer(meta_features.to(dtype=packed_logits.dtype))
            )
            if mode == "rich_bilinear":
                hidden = torch.tanh(candidate_features)
                residual = (hidden * context_features).sum(dim=-1).to(dtype=packed_logits.dtype) * scale
            else:
                hidden = torch.tanh(candidate_features + context_features)
                residual = out_layer(hidden).squeeze(-1).to(dtype=packed_logits.dtype) * scale
        elif mode == "bilinear":
            candidate_features = state_layer(row_state) + meta_layer(meta_features.to(dtype=packed_logits.dtype))
            hidden = torch.tanh(candidate_features)
            residual = (hidden * context_features).sum(dim=-1).to(dtype=packed_logits.dtype) * scale
        else:
            candidate_features = state_layer(row_state) + meta_layer(meta_features.to(dtype=packed_logits.dtype))
            hidden = torch.tanh(candidate_features + context_features)
            residual = out_layer(hidden).squeeze(-1).to(dtype=packed_logits.dtype) * scale
        allowed_action_ids = tuple(
            int(action_id) for action_id in getattr(self, "opponent_context_candidate_residual_action_ids", ())
        )
        if allowed_action_ids:
            if legal_actions.ids is None:
                return packed_logits
            action_ids = torch.as_tensor(legal_actions.ids, device=packed_logits.device, dtype=torch.long)
            if int(action_ids.numel()) != int(packed_logits.shape[0]):
                raise ValueError("packed legal ids must align with packed logits")
            allowed = torch.zeros_like(action_ids, dtype=torch.bool)
            for action_id in allowed_action_ids:
                allowed |= action_ids == int(action_id)
            residual = residual.masked_fill(~allowed, 0.0)
        residual = residual.masked_fill(row_context == 0, 0.0)
        return packed_logits + residual

    def _apply_opponent_context_packed_candidate_residual_to_log_probs(
        self,
        packed_log_probs: Tensor,
        legal_actions: LegalActionBatch,
        state_repr: Tensor,
        opponent_context_index: Tensor | None,
        *,
        observation_context: Mapping[str, Tensor] | None = None,
        scoring_mode: str = "auto",
    ) -> Tensor:
        if legal_actions.offsets is None or legal_actions.meta is None:
            return packed_log_probs
        biased = self._apply_opponent_context_packed_candidate_residual(
            packed_log_probs,
            legal_actions,
            state_repr,
            opponent_context_index,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )
        if biased is packed_log_probs:
            return packed_log_probs
        if biased.ndim != 1:
            raise ValueError(f"packed_log_probs must be 1D, got shape {tuple(biased.shape)}")
        offsets = torch.as_tensor(legal_actions.offsets, device=biased.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.sum().item()) != int(biased.shape[0]):
            raise ValueError("packed legal offsets must align with packed log-probs")
        if int(biased.shape[0]) == 0:
            return biased
        row_count = int(offsets.numel() - 1)
        row_indices = torch.repeat_interleave(torch.arange(row_count, device=biased.device), lengths)
        row_log_z = self._segment_logsumexp_1d(biased, row_indices, row_count)
        return biased - row_log_z.index_select(0, row_indices)

    def _apply_opponent_context_packed_action_bias_to_log_probs(
        self,
        packed_log_probs: Tensor,
        legal_actions: LegalActionBatch,
        opponent_context_index: Tensor | None,
    ) -> Tensor:
        if legal_actions.ids is None or legal_actions.offsets is None:
            return packed_log_probs
        biased = self._apply_opponent_context_packed_action_bias(
            packed_log_probs,
            legal_actions,
            opponent_context_index,
        )
        if biased is packed_log_probs:
            return packed_log_probs
        if biased.ndim != 1:
            raise ValueError(f"packed_log_probs must be 1D, got shape {tuple(biased.shape)}")
        offsets = torch.as_tensor(legal_actions.offsets, device=biased.device, dtype=torch.long)
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.sum().item()) != int(biased.shape[0]):
            raise ValueError("packed legal offsets must align with packed log-probs")
        if int(biased.shape[0]) == 0:
            return biased
        row_count = int(offsets.numel() - 1)
        row_indices = torch.repeat_interleave(torch.arange(row_count, device=biased.device), lengths)
        row_log_z = self._segment_logsumexp_1d(biased, row_indices, row_count)
        return biased - row_log_z.index_select(0, row_indices)

    def _segment_logsumexp_1d(self, values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
        out_max = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
        if keys.numel() == 0:
            return out_max
        long_keys = keys.to(dtype=torch.long)
        out_max.scatter_reduce_(0, long_keys, values, reduce="amax", include_self=True)
        gathered_max = out_max.index_select(0, long_keys)
        shifted = torch.exp(values - gathered_max)
        sumexp = torch.zeros((int(num_segments),), dtype=values.dtype, device=values.device)
        sumexp.scatter_add_(0, long_keys, shifted)
        valid = torch.isfinite(out_max) & (sumexp > 0)
        result = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
        result[valid] = torch.log(sumexp[valid]) + out_max[valid]
        return result

    def _prepare_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        return prepare_hidden_state(
            hidden_state,
            batch_size=batch_size,
            like=like,
            hidden_size=self.hidden_size,
            initial_hidden=lambda current_batch_size: self.initial_hidden(
                current_batch_size,
                device=like.device,
                dtype=like.dtype,
            ),
        )

    def _prepare_seat_hidden_state(self, hidden_state: Tensor | None, *, batch_size: int, like: Tensor) -> Tensor:
        return prepare_seat_hidden_state(
            hidden_state,
            batch_size=batch_size,
            like=like,
            hidden_size=self.hidden_size,
            seat_count=SEAT_COUNT,
            initial_seat_hidden=lambda current_batch_size: self.initial_seat_hidden(
                current_batch_size,
                device=like.device,
                dtype=like.dtype,
            ),
        )

    def _prepare_acting_seat(self, acting_seat: int | Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        return prepare_acting_seat(acting_seat, batch_size=batch_size, device=device)

    def _select_acting_hidden(self, seat_hidden_state: Tensor, acting_seat: Tensor) -> Tensor:
        return select_acting_hidden(seat_hidden_state, acting_seat, hidden_size=self.hidden_size)

    def _write_acting_hidden(
        self,
        seat_hidden_state: Tensor,
        acting_seat: Tensor,
        next_acting_hidden: Tensor,
    ) -> Tensor:
        return write_acting_hidden(seat_hidden_state, acting_seat, next_acting_hidden)

    def _reference_parameter(self: Any) -> Tensor:
        try:
            return next(self.parameters())
        except StopIteration as exc:
            raise RuntimeError("Model has no parameters to use as a reference tensor") from exc
