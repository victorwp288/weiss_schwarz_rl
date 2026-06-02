"""Policy-row application helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.masking import (
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
    select_argmax_from_legal_ids,
    select_argmax_from_mask,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, _pack_batch
from weiss_rl.runtime.components import shared as runtime_shared
from weiss_rl.runtime.components.legal_batching import (
    slice_packed_rows,
    structured_legal_batch_from_mask,
    structured_legal_batch_from_packed,
)
from weiss_rl.runtime.components.legal_meta import ensure_legal_action_meta, legal_action_meta_from_ids
from weiss_rl.runtime.components.opponent_context import _call_accepts_keyword
from weiss_rl.runtime.components.policy_inference.debug_validation import (
    validate_env_step_packed_actions,
    validate_sampled_packed_actions,
)
from weiss_rl.runtime.components.policy_inference.deterministic_logits import write_deterministic_logits_from_packed

_DEFAULT_ACTION_META_WIDTH = runtime_shared.DEFAULT_ACTION_META_WIDTH

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState
    from weiss_rl.runtime.components.config import QueueRuntimeConfig


class QueueRuntimePolicyRowsMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device
        config: QueueRuntimeConfig

    def _apply_policy_rows_mask(
        self,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_mask: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
        action_selection: str = "sample",
        source_label: str = "policy_rows",
    ) -> None:
        del source_label
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            legal_actions = (
                structured_legal_batch_from_mask(legal_mask, row_indices)
                if bool(getattr(model, "supports_legal_candidate_scoring", False))
                else None
            )
            logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                torch.as_tensor(obs_step[row_indices], device=self._device),
                torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                hidden_state[row_indices],
                legal_actions=legal_actions,
            )
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()
        logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits_out is not None:
            logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        if sample_actions:
            assert actions_out is not None and logp_out is not None
            if str(action_selection) == "argmax":
                action_subset = select_argmax_from_mask(
                    logits_subset,
                    legal_mask[row_indices],
                    pass_action_id=self.config.pass_action_id,
                )
                logp_subset = np.zeros((row_indices.shape[0],), dtype=np.float32)
            else:
                action_subset, logp_subset, _entropy = sample_actions_from_mask(
                    logits_subset,
                    legal_mask[row_indices],
                    rng=rng,
                    pass_action_id=self.config.pass_action_id,
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                )
            actions_out[row_indices] = action_subset
            logp_out[row_indices] = logp_subset

    def _maybe_debug_validate_sampled_packed_actions(
        self,
        *,
        source_label: str,
        row_indices: np.ndarray,
        action_subset: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> None:
        if not bool(getattr(self, "_debug_validate_sampled_packed_actions", False)):
            return
        validate_sampled_packed_actions(
            source_label=source_label,
            row_indices=row_indices,
            action_subset=action_subset,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            pass_action_id=int(self.config.pass_action_id),
        )

    def _maybe_debug_validate_env_step_packed_actions(
        self,
        *,
        actor: _ActorState,
        source_label: str,
        actions: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> None:
        if not bool(getattr(self, "_debug_validate_sampled_packed_actions", False)):
            return
        validate_env_step_packed_actions(
            source_label=source_label,
            actions=actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            env_batch=getattr(actor.env, "_last_batch", None),
        )

    def _sync_actor_batch_from_step_out(
        self,
        *,
        actor: _ActorState,
        step_out: Any,
        pool: Any,
    ) -> DecisionBoundaryBatch:
        batch = _pack_batch(
            step_out,
            legality="ids_offsets",
            pool=pool,
            copy_arrays=False,
        )
        if batch.ids_offsets is not None and batch.legal_action_meta is None:
            legal_action_meta = self._legal_action_meta_from_ids(batch.ids_offsets[0])
            if legal_action_meta is not None:
                batch = replace(batch, legal_action_meta=legal_action_meta)
        actor.current_batch = batch
        actor.env._last_batch = batch
        return batch

    def _legal_action_meta_from_ids(self, legal_ids: np.ndarray) -> np.ndarray | None:
        return legal_action_meta_from_ids(
            legal_ids,
            action_catalog=getattr(self, "_action_catalog", None),
            family_index=getattr(self, "_action_family_index", {}),
            attack_type_index=getattr(self, "_action_attack_type_index", {}),
            action_meta_width=int(getattr(self, "_action_meta_width", _DEFAULT_ACTION_META_WIDTH)),
        )

    def _ensure_legal_action_meta(
        self,
        legal_ids: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray | None:
        return ensure_legal_action_meta(
            legal_ids,
            legal_action_meta,
            build_meta=self._legal_action_meta_from_ids,
        )

    def _apply_policy_rows_ids(
        self,
        *,
        model: Any,
        hidden_state: torch.Tensor,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        opponent_context_index: np.ndarray | None = None,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
        action_selection: str = "sample",
        source_label: str = "policy_rows",
    ) -> None:
        action_selection = str(action_selection)
        supports_structured_candidates = bool(getattr(model, "supports_legal_candidate_scoring", False))
        structured_meta = (
            self._ensure_legal_action_meta(legal_ids, legal_action_meta) if supports_structured_candidates else None
        )
        legal_actions = (
            structured_legal_batch_from_packed(
                legal_ids,
                legal_offsets,
                row_indices,
                structured_meta,
            )
            if supports_structured_candidates
            else None
        )
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            context_tensor = (
                None
                if opponent_context_index is None
                else torch.as_tensor(opponent_context_index[row_indices], device=self._device, dtype=torch.long)
            )
            if (
                legal_actions is not None
                and sample_actions
                and action_selection == "sample"
                and logits_out is None
                and bool(getattr(model, "supports_factorized_legal_policy", False))
                and hasattr(model, "sample_factorized_packed_seat_aware")
            ):
                action_tensor, logp_tensor, value_tensor, next_hidden = model.sample_factorized_packed_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    legal_actions=legal_actions,
                    sample_seeds=torch.as_tensor(
                        rng.integers(0, np.iinfo(np.int64).max, size=row_indices.shape[0], dtype=np.int64),
                        device=self._device,
                        dtype=torch.long,
                    ),
                    pass_action_id=int(self.config.pass_action_id),
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    **(
                        {"opponent_context_index": context_tensor}
                        if _call_accepts_keyword(model.sample_factorized_packed_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
                logits_subset = None
            elif (
                legal_actions is not None
                and sample_actions
                and action_selection == "sample"
                and logits_out is None
                and hasattr(model, "sample_packed_seat_aware")
            ):
                action_tensor, logp_tensor, value_tensor, next_hidden = model.sample_packed_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    legal_actions=legal_actions,
                    sample_seeds=torch.as_tensor(
                        rng.integers(0, np.iinfo(np.int64).max, size=row_indices.shape[0], dtype=np.int64),
                        device=self._device,
                        dtype=torch.long,
                    ),
                    pass_action_id=int(self.config.pass_action_id),
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                    **(
                        {"opponent_context_index": context_tensor}
                        if _call_accepts_keyword(model.sample_packed_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
                logits_subset = None
            elif legal_actions is None:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    **(
                        {"opponent_context_index": context_tensor}
                        if _call_accepts_keyword(model.forward_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
                logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_step[row_indices], device=self._device),
                    torch.as_tensor(actor_step[row_indices], device=self._device, dtype=torch.long),
                    hidden_state[row_indices],
                    legal_actions=legal_actions,
                    **(
                        {"opponent_context_index": context_tensor}
                        if _call_accepts_keyword(model.forward_seat_aware, "opponent_context_index")
                        else {}
                    ),
                )
                logits_subset = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        hidden_state[row_indices] = torch.as_tensor(
            next_hidden,
            device=self._device,
            dtype=hidden_state.dtype,
        ).clone()
        value_subset = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if logits_out is not None:
            assert logits_subset is not None
            logits_out[row_indices] = logits_subset
        values_out[row_indices] = value_subset
        if action_selection == "argmax" and logits_subset is not None and logits_out is not None:
            action_subset = select_argmax_from_legal_ids(
                logits_subset,
                *slice_packed_rows(legal_ids, legal_offsets, row_indices),
                pass_action_id=self.config.pass_action_id,
            )
            write_deterministic_logits_from_packed(
                logits_out=logits_out,
                row_indices=row_indices,
                chosen_actions=action_subset,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
            )
        if sample_actions:
            assert actions_out is not None and logp_out is not None
            if logits_subset is None:
                action_subset = action_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
                logp_subset = logp_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            elif action_selection == "argmax":
                subset_ids, subset_offsets = slice_packed_rows(legal_ids, legal_offsets, row_indices)
                action_subset = select_argmax_from_legal_ids(
                    logits_subset,
                    subset_ids,
                    subset_offsets,
                    pass_action_id=self.config.pass_action_id,
                )
                logp_subset = np.zeros((row_indices.shape[0],), dtype=np.float32)
            else:
                subset_ids, subset_offsets = slice_packed_rows(legal_ids, legal_offsets, row_indices)
                action_subset, logp_subset, _entropy = sample_actions_from_legal_ids(
                    logits_subset,
                    subset_ids,
                    subset_offsets,
                    rng=rng,
                    pass_action_id=self.config.pass_action_id,
                    temperature=float(getattr(self.config, "actor_sampling_temperature", 1.0)),
                )
            self._maybe_debug_validate_sampled_packed_actions(
                source_label=source_label,
                row_indices=row_indices,
                action_subset=np.asarray(action_subset, dtype=np.int64),
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
            )
            actions_out[row_indices] = action_subset
            logp_out[row_indices] = logp_subset
