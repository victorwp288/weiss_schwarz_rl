"""Central opponent-output overwrite helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.masking import select_argmax_from_legal_ids, select_argmax_from_mask
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.policy_set import heuristic_public_profile_name_for_policy_id
from weiss_rl.runtime_components.batching import (
    optional_legal_action_meta,
    require_ids_offsets,
    require_mask,
    slice_packed_rows_with_meta,
)
from weiss_rl.runtime_components.policy_ids import MIRROR_OPPONENT_POLICY_ID

if TYPE_CHECKING:
    from weiss_rl.runtime_components.actor_state import _ActorState


class QueueRuntimeCentralOpponentMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _overwrite_central_outputs_with_configured_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "python_scalar":
            for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
                actors,
                batches,
                obs_steps,
                actor_steps,
                logits_outs,
                values_outs,
                strict=True,
            ):
                self._overwrite_central_outputs_with_batched_opponents(
                    actors=[actor],
                    batches=[batch],
                    obs_steps=[obs_step],
                    actor_steps=[actor_step],
                    logits_outs=[logits_out],
                    values_outs=[values_out],
                )
            return
        self._overwrite_central_outputs_with_batched_opponents(
            actors=actors,
            batches=batches,
            obs_steps=obs_steps,
            actor_steps=actor_steps,
            logits_outs=logits_outs,
            values_outs=values_outs,
        )

    def _overwrite_central_outputs_with_opponents(
        self,
        *,
        actor: _ActorState,
        batch: DecisionBoundaryBatch,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
    ) -> None:
        self._overwrite_central_outputs_with_configured_opponents(
            actors=[actor],
            batches=[batch],
            obs_steps=[obs_step],
            actor_steps=[actor_step],
            logits_outs=[logits_out],
            values_outs=[values_out],
        )

    def _overwrite_central_outputs_with_batched_opponents(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray | None],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        import time

        overwrite_started = time.perf_counter()
        policy_groups: dict[
            str,
            list[
                tuple[
                    _ActorState,
                    DecisionBoundaryBatch,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray | None,
                    np.ndarray,
                ]
            ],
        ] = {}
        for actor, batch, obs_step, actor_step, logits_out, values_out in zip(
            actors,
            batches,
            obs_steps,
            actor_steps,
            logits_outs,
            values_outs,
            strict=True,
        ):
            focal_rows = actor_step == actor.focal_seat_by_env
            opponent_indices = np.flatnonzero(~focal_rows)
            if opponent_indices.size == 0:
                continue
            for policy_id in sorted(
                {str(actor.opponent_policy_id_by_env[index]) for index in opponent_indices.tolist()}
            ):
                if policy_id == MIRROR_OPPONENT_POLICY_ID:
                    continue
                policy_rows = opponent_indices[actor.opponent_policy_id_by_env[opponent_indices] == policy_id]
                if not policy_rows.size:
                    continue
                policy_groups.setdefault(policy_id, []).append(
                    (actor, batch, policy_rows, obs_step, actor_step, logits_out, values_out)
                )

        for policy_id, entries in sorted(policy_groups.items()):
            heuristic_policy = self._heuristic_opponent_policy(policy_id)
            if heuristic_policy is not None:
                if self._should_track_heuristic_actor_hidden_state():
                    self._central_advance_actor_rows(
                        actors=[actor for actor, *_rest in entries],
                        obs_steps=[
                            obs_step
                            for _actor, _batch, _row_indices, obs_step, _actor_step, _logits_out, _values_out in entries
                        ],
                        actor_steps=[
                            actor_step
                            for _actor, _batch, _row_indices, _obs_step, actor_step, _logits_out, _values_out in entries
                        ],
                        row_indices_by_actor=[
                            row_indices
                            for _actor, _batch, row_indices, _obs_step, _actor_step, _logits_out, _values_out in entries
                        ],
                    )
                packed_entries = [entry for entry in entries if entry[1].ids_offsets is not None]
                mask_entries = [entry for entry in entries if entry[1].ids_offsets is None]
                if packed_entries:
                    if str(getattr(self, "_fixed_opponent_backend", "python_batched")) == "simulator_native":
                        for actor, batch, row_indices, obs_step, _actor_step, logits_out, values_out in packed_entries:
                            legal_ids, legal_offsets = require_ids_offsets(batch)
                            chosen_actions = self._heuristic_public_actions_from_ids(
                                actor=actor,
                                heuristic_policy=heuristic_policy,
                                row_indices=row_indices,
                                obs_step=obs_step,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                                legal_action_meta=self._ensure_legal_action_meta(
                                    legal_ids, optional_legal_action_meta(batch)
                                ),
                                profile_name=heuristic_public_profile_name_for_policy_id(policy_id),
                            )
                            self._maybe_debug_validate_sampled_packed_actions(
                                source_label=f"central:opponent:{policy_id}:heuristic",
                                row_indices=row_indices,
                                action_subset=np.asarray(chosen_actions, dtype=np.int64),
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                            self._write_deterministic_logits_from_packed(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                            values_out[row_indices] = 0.0
                    else:
                        packed_obs_parts: list[np.ndarray] = []
                        packed_ids: list[np.ndarray] = []
                        packed_meta: list[np.ndarray] = []
                        packed_offsets = [np.array([0], dtype=np.uint32)]
                        packed_entry_counts: list[int] = []
                        for (
                            _actor,
                            batch,
                            row_indices,
                            obs_step,
                            _actor_step,
                            _logits_out,
                            _values_out,
                        ) in packed_entries:
                            legal_ids, legal_offsets = require_ids_offsets(batch)
                            subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
                                legal_ids,
                                legal_offsets,
                                row_indices,
                                legal_action_meta=self._ensure_legal_action_meta(
                                    legal_ids, optional_legal_action_meta(batch)
                                ),
                            )
                            offset_base = int(packed_offsets[-1][-1])
                            packed_ids.append(subset_ids)
                            packed_offsets.append(np.asarray(subset_offsets[1:] + offset_base, dtype=np.uint32))
                            if subset_meta is not None:
                                packed_meta.append(subset_meta)
                            packed_obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.int32))
                            packed_entry_counts.append(int(row_indices.shape[0]))
                        packed_chosen_actions = heuristic_policy.choose_actions_from_meta_batch(
                            np.concatenate(packed_obs_parts, axis=0)
                            if packed_obs_parts
                            else np.zeros((0, 0), dtype=np.int32),
                            np.concatenate(packed_ids, axis=0) if packed_ids else np.zeros((0,), dtype=np.uint32),
                            np.concatenate(packed_offsets, axis=0),
                            np.concatenate(packed_meta, axis=0) if packed_meta else None,
                        )
                        offset = 0
                        for (_actor, batch, row_indices, _obs_step, _actor_step, logits_out, values_out), count in zip(
                            packed_entries,
                            packed_entry_counts,
                            strict=True,
                        ):
                            legal_ids, legal_offsets = require_ids_offsets(batch)
                            chosen_actions = np.asarray(
                                packed_chosen_actions[offset : offset + count],
                                dtype=np.int64,
                            )
                            self._write_deterministic_logits_from_packed(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                            values_out[row_indices] = 0.0
                            offset += count
                for actor, batch, row_indices, obs_step, _actor_step, logits_out, values_out in mask_entries:
                    legal_mask = require_mask(batch)
                    chosen_actions = self._heuristic_public_actions_from_mask(
                        actor=actor,
                        heuristic_policy=heuristic_policy,
                        row_indices=row_indices,
                        obs_step=obs_step,
                        legal_mask=legal_mask,
                        profile_name=heuristic_public_profile_name_for_policy_id(policy_id),
                    )
                    legal_action_ids = [
                        np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(
                            np.uint32,
                            copy=False,
                        )
                        for row_index in row_indices.tolist()
                    ]
                    self._write_deterministic_logits(
                        logits_out=logits_out,
                        row_indices=row_indices,
                        chosen_actions=chosen_actions,
                        legal_action_ids=legal_action_ids,
                    )
                    values_out[row_indices] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            obs_concat = np.concatenate(
                [obs_step[row_indices] for _, _, row_indices, obs_step, _, _, _ in entries],
                axis=0,
            )
            actor_concat = np.concatenate(
                [actor_step[row_indices] for _, _, row_indices, _, actor_step, _, _ in entries],
                axis=0,
            )
            hidden_concat = torch.cat(
                [actor.opponent_hidden[row_indices] for actor, _, row_indices, _, _, _, _ in entries],
                dim=0,
            )
            with (
                self._opponent_model_locks[policy_id],
                torch.inference_mode(),
                torch.amp.autocast(
                    device_type=self._device.type,
                    enabled=self._actor_amp_enabled,
                ),
            ):
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(obs_concat, device=self._device),
                    torch.as_tensor(actor_concat, device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            logits_concat = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
            offset = 0
            for actor, _batch, row_indices, _obs_step, _actor_step, logits_out, values_out in entries:
                count = int(row_indices.shape[0])
                actor.opponent_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
                values_out[row_indices] = values_concat[offset : offset + count]
                if logits_out is not None:
                    row_logits = logits_concat[offset : offset + count]
                    logits_out[row_indices] = row_logits
                    config = getattr(self, "config", None)
                    if str(getattr(config, "fixed_model_opponent_action_selection", "sample")) == "argmax":
                        if _batch.ids_offsets is not None:
                            legal_ids, legal_offsets = require_ids_offsets(_batch)
                            subset_ids, subset_offsets, _subset_meta = slice_packed_rows_with_meta(
                                legal_ids,
                                legal_offsets,
                                row_indices,
                                legal_action_meta=self._ensure_legal_action_meta(
                                    legal_ids, optional_legal_action_meta(_batch)
                                ),
                            )
                            chosen_actions = select_argmax_from_legal_ids(
                                row_logits,
                                subset_ids,
                                subset_offsets,
                                pass_action_id=self.config.pass_action_id,
                            )
                            self._write_deterministic_logits_from_packed(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_ids=legal_ids,
                                legal_offsets=legal_offsets,
                            )
                        else:
                            legal_mask = require_mask(_batch)
                            chosen_actions = select_argmax_from_mask(
                                row_logits,
                                legal_mask[row_indices],
                                pass_action_id=self.config.pass_action_id,
                            )
                            legal_action_ids = [
                                np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(
                                    np.uint32,
                                    copy=False,
                                )
                                for row_index in row_indices.tolist()
                            ]
                            self._write_deterministic_logits(
                                logits_out=logits_out,
                                row_indices=row_indices,
                                chosen_actions=chosen_actions,
                                legal_action_ids=legal_action_ids,
                            )
                offset += count
        self._record_batch_timer_ms("central_fixed_opponent_overwrite", time.perf_counter() - overwrite_started)
