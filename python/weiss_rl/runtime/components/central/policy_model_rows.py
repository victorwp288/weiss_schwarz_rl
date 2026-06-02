"""Model-backed central packed policy-row sampling."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.central.legal_hooks import (
    optional_legal_action_meta,
    require_ids_offsets,
    slice_packed_rows_with_meta,
)
from weiss_rl.runtime.components.central.model_hooks import actor_inference_model
from weiss_rl.runtime.components.opponent_context import _call_accepts_keyword, opponent_context_indices_for_model

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


class QueueRuntimeCentralPolicyModelRowsMixin:
    def _central_sample_policy_rows_ids_model(
        self: Any,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
        actions_outs: Sequence[np.ndarray],
        logp_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, _ActorState, np.ndarray]] = []
        packed_ids: list[np.ndarray] = []
        packed_meta: list[np.ndarray] = []
        packed_offsets = [np.array([0], dtype=np.uint32)]
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        context_parts: list[np.ndarray] = []
        seed_parts: list[np.ndarray] = []
        model = actor_inference_model(actors[0])
        pack_started = time.perf_counter()
        model_row_count = 0
        for actor_index, (actor, batch, obs_step, actor_step, row_indices) in enumerate(
            zip(
                actors,
                batches,
                obs_steps,
                actor_steps,
                row_indices_by_actor,
                strict=True,
            )
        ):
            if row_indices.size == 0:
                continue
            model_row_count += int(row_indices.size)
            legal_ids, legal_offsets = require_ids_offsets(batch)
            legal_action_meta = self._ensure_legal_action_meta(legal_ids, optional_legal_action_meta(batch))
            subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
                legal_ids,
                legal_offsets,
                row_indices,
                legal_action_meta=legal_action_meta,
            )
            offset_base = int(packed_offsets[-1][-1])
            packed_ids.append(subset_ids)
            packed_offsets.append(np.asarray(subset_offsets[1:] + offset_base, dtype=np.uint32))
            if subset_meta is not None:
                packed_meta.append(subset_meta)
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            context_parts.append(
                opponent_context_indices_for_model(
                    model,
                    actor.opponent_policy_id_by_env[row_indices],
                    batch_size=int(row_indices.shape[0]),
                )
            )
            seed_parts.append(actor.rng.integers(0, np.iinfo(np.int64).max, size=row_indices.shape[0], dtype=np.int64))
            entries.append((actor_index, actor, row_indices))
        if not entries:
            return
        legal_actions = LegalActionBatch.from_packed(
            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
            np.concatenate(packed_offsets, axis=0),
            meta=(np.concatenate(packed_meta, axis=0) if packed_meta else None),
            action_space=int(self.action_dim),
        )
        hidden_concat = torch.cat(hidden_parts, dim=0)
        self._record_batch_counter("central_focal_policy_model_rows", float(model_row_count))
        self._record_batch_counter(
            "central_focal_policy_model_candidates",
            0.0 if legal_actions.ids is None else float(legal_actions.ids.size),
        )
        self._record_batch_timer_ms("central_focal_policy_pack", time.perf_counter() - pack_started)
        model_started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            context_concat = torch.as_tensor(
                np.concatenate(context_parts, axis=0),
                device=self._device,
                dtype=torch.long,
            )
            if bool(getattr(model, "supports_factorized_legal_policy", False)) and hasattr(
                model,
                "sample_factorized_packed_seat_aware",
            ):
                actions_tensor, logp_tensor, value_tensor, next_hidden = model.sample_factorized_packed_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                    legal_actions=legal_actions,
                    sample_seeds=torch.as_tensor(
                        np.concatenate(seed_parts, axis=0), device=self._device, dtype=torch.long
                    ),
                    pass_action_id=int(self.config.pass_action_id),
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    **(
                        {"opponent_context_index": context_concat}
                        if _call_accepts_keyword(model.sample_factorized_packed_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
            else:
                actions_tensor, logp_tensor, value_tensor, next_hidden = model.sample_packed_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                    legal_actions=legal_actions,
                    sample_seeds=torch.as_tensor(
                        np.concatenate(seed_parts, axis=0), device=self._device, dtype=torch.long
                    ),
                    pass_action_id=int(self.config.pass_action_id),
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    **(
                        {"opponent_context_index": context_concat}
                        if _call_accepts_keyword(model.sample_packed_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
        self._record_batch_timer_ms("central_focal_policy_model", time.perf_counter() - model_started)
        actions_concat = actions_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        logp_concat = logp_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        scatter_started = time.perf_counter()
        offset = 0
        for actor_index, actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            actions_outs[actor_index][row_indices] = actions_concat[offset : offset + count]
            logp_outs[actor_index][row_indices] = logp_concat[offset : offset + count]
            offset += count
        self._record_batch_timer_ms("central_focal_policy_scatter", time.perf_counter() - scatter_started)


__all__ = ["QueueRuntimeCentralPolicyModelRowsMixin"]
