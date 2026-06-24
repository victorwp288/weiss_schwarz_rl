"""Opponent-context adapters used by policy/value model forward paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import cast

import torch
from torch import Tensor, nn

from weiss_rl.config.models import ModelConfig
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.models.policy.opponent_context import build_opponent_context_offsets
from weiss_rl.models.policy.opponent_context_packed import (
    apply_packed_action_bias,
    apply_packed_action_bias_to_log_probs,
    apply_packed_candidate_residual,
    apply_packed_candidate_residual_to_log_probs,
    segment_logsumexp_1d,
)


class PolicyValueModelOpponentContextMixin:
    """Maps opponent identities to hidden offsets, action bias, and candidate residuals."""

    hidden_size: int

    def _configure_opponent_context(self, *, config: ModelConfig, action_dim: int) -> None:
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
            build_opponent_context_offsets(
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

    def _install_candidate_residual_adapter(self, *, config: ModelConfig) -> None:
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
        return apply_packed_action_bias(
            self,
            packed_logits,
            legal_actions,
            opponent_context_index,
        )

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
        return apply_packed_candidate_residual(
            self,
            packed_logits,
            legal_actions,
            state_repr,
            opponent_context_index,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )

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
        return apply_packed_candidate_residual_to_log_probs(
            self,
            packed_log_probs,
            legal_actions,
            state_repr,
            opponent_context_index,
            observation_context=observation_context,
            scoring_mode=scoring_mode,
        )

    def _apply_opponent_context_packed_action_bias_to_log_probs(
        self,
        packed_log_probs: Tensor,
        legal_actions: LegalActionBatch,
        opponent_context_index: Tensor | None,
    ) -> Tensor:
        return apply_packed_action_bias_to_log_probs(
            self,
            packed_log_probs,
            legal_actions,
            opponent_context_index,
        )

    def _segment_logsumexp_1d(self, values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
        return segment_logsumexp_1d(values, keys, num_segments)


__all__ = ["PolicyValueModelOpponentContextMixin"]
