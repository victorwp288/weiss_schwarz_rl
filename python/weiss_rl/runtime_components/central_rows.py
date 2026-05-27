"""Central actor-row forwarding helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime_components.heuristic_actor_outputs import write_heuristic_actor_outputs_ids
from weiss_rl.runtime_components.opponent_context import _call_accepts_keyword, opponent_context_indices_for_model

if TYPE_CHECKING:
    from weiss_rl.runtime_components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


def _concatenate_batch_legal_actions(batches: Sequence[DecisionBoundaryBatch], *, action_space: int) -> Any:
    # Resolve lazily through weiss_rl.runtime so the compatibility wrapper remains the public hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._concatenate_batch_legal_actions(batches, action_space=action_space)


def _model_accepts_legal_actions_kwarg(model: Any) -> bool:
    target = getattr(model, "_orig_mod", model)
    forward = getattr(target, "forward_seat_aware", None)
    if forward is None:
        return False
    try:
        parameters = signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if "legal_actions" in parameters:
        return True
    return any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())


def _optional_legal_action_meta(batch: DecisionBoundaryBatch) -> np.ndarray | None:
    from weiss_rl import runtime as runtime_module

    return runtime_module._optional_legal_action_meta(batch)


def _require_ids_offsets(batch: DecisionBoundaryBatch) -> tuple[np.ndarray, np.ndarray]:
    from weiss_rl import runtime as runtime_module

    return runtime_module._require_ids_offsets(batch)


def _slice_packed_rows_with_meta(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    row_indices: np.ndarray,
    *,
    legal_action_meta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from weiss_rl import runtime as runtime_module

    return runtime_module._slice_packed_rows_with_meta(
        legal_ids,
        legal_offsets,
        row_indices,
        legal_action_meta=legal_action_meta,
    )


class QueueRuntimeCentralRowsMixin:
    if TYPE_CHECKING:
        _actor_amp_enabled: bool
        _device: torch.device
        action_dim: int

    def _central_value_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = _actor_inference_model(actors[0])
        for actor_index, (actor, obs_step, actor_step, row_indices) in enumerate(
            zip(actors, obs_steps, actor_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor_index, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            value_seat_aware = getattr(model, "value_seat_aware", None)
            if callable(value_seat_aware):
                value_tensor = value_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, value_tensor, _next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        offset = 0
        for actor_index, row_indices in entries:
            count = int(row_indices.shape[0])
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            offset += count

    def _central_value_and_advance_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[int, _ActorState, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = _actor_inference_model(actors[0])
        for actor_index, (actor, obs_step, actor_step, row_indices) in enumerate(
            zip(actors, obs_steps, actor_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor_index, actor, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        obs_tensor = torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device)
        actor_tensor = torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            forward_trunk = getattr(model, "forward_trunk_packed_seat_aware", None)
            if callable(forward_trunk):
                _recurrent_output, _state_repr, _observation_context, value_tensor, next_hidden = forward_trunk(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                )
            else:
                value_seat_aware = getattr(model, "value_seat_aware", None)
                advance_hidden = getattr(model, "advance_seat_hidden", None)
                if callable(value_seat_aware) and callable(advance_hidden):
                    value_tensor = value_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
                    next_hidden = advance_hidden(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
                else:
                    _logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                        obs_tensor,
                        actor_tensor,
                        hidden_concat,
                    )
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor_index, actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            values_outs[actor_index][row_indices] = values_concat[offset : offset + count]
            offset += count

    def _central_advance_actor_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        row_indices_by_actor: Sequence[np.ndarray],
    ) -> None:
        entries: list[tuple[_ActorState, np.ndarray]] = []
        obs_parts: list[np.ndarray] = []
        actor_parts: list[np.ndarray] = []
        hidden_parts: list[torch.Tensor] = []
        model = _actor_inference_model(actors[0])
        for actor, obs_step, actor_step, row_indices in zip(
            actors,
            obs_steps,
            actor_steps,
            row_indices_by_actor,
            strict=True,
        ):
            if row_indices.size == 0:
                continue
            obs_parts.append(np.asarray(obs_step[row_indices], dtype=np.float32))
            actor_parts.append(np.asarray(actor_step[row_indices], dtype=np.int64))
            hidden_parts.append(actor.seat_hidden[row_indices])
            entries.append((actor, row_indices))
        if not entries:
            return
        hidden_concat = torch.cat(hidden_parts, dim=0)
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            advance_hidden = getattr(model, "advance_seat_hidden", None)
            if callable(advance_hidden):
                next_hidden = advance_hidden(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
            else:
                _logits_tensor, _value_tensor, next_hidden = model.forward_seat_aware(
                    torch.as_tensor(np.concatenate(obs_parts, axis=0), device=self._device),
                    torch.as_tensor(np.concatenate(actor_parts, axis=0), device=self._device, dtype=torch.long),
                    hidden_concat,
                )
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, row_indices in entries:
            count = int(row_indices.shape[0])
            actor.seat_hidden[row_indices] = next_hidden_tensor[offset : offset + count]
            offset += count

    def _central_forward_all_rows(
        self,
        *,
        actors: Sequence[_ActorState],
        batches: Sequence[DecisionBoundaryBatch] | None,
        obs_steps: Sequence[np.ndarray],
        actor_steps: Sequence[np.ndarray],
        logits_outs: Sequence[np.ndarray],
        values_outs: Sequence[np.ndarray],
    ) -> None:
        if not actors:
            return
        obs_concat = np.concatenate(obs_steps, axis=0)
        actor_concat = np.concatenate(actor_steps, axis=0)
        hidden_concat = torch.cat([actor.seat_hidden for actor in actors], dim=0)
        model = _actor_inference_model(actors[0])
        legal_actions = None
        if (
            bool(getattr(model, "supports_legal_candidate_scoring", False))
            and batches is not None
            and _model_accepts_legal_actions_kwarg(model)
        ):
            legal_actions = _concatenate_batch_legal_actions(batches, action_space=int(self.action_dim))
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                device_type=self._device.type,
                enabled=self._actor_amp_enabled,
            ),
        ):
            obs_tensor = torch.as_tensor(obs_concat, device=self._device)
            actor_tensor = torch.as_tensor(actor_concat, device=self._device, dtype=torch.long)
            if legal_actions is None:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                )
            else:
                logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
                    obs_tensor,
                    actor_tensor,
                    hidden_concat,
                    legal_actions=legal_actions,
                )
        logits_concat = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        values_concat = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        next_hidden_tensor = torch.as_tensor(next_hidden, device=self._device, dtype=hidden_concat.dtype)
        offset = 0
        for actor, logits_out, values_out in zip(actors, logits_outs, values_outs, strict=True):
            count = int(logits_out.shape[0])
            logits_out[...] = logits_concat[offset : offset + count]
            values_out[...] = values_concat[offset : offset + count]
            actor.seat_hidden[...] = next_hidden_tensor[offset : offset + count]
            offset += count

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
        model = _actor_inference_model(actors[0])
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
            legal_ids, legal_offsets = _require_ids_offsets(batch)
            legal_action_meta = self._ensure_legal_action_meta(legal_ids, _optional_legal_action_meta(batch))
            subset_ids, subset_offsets, subset_meta = _slice_packed_rows_with_meta(
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

    def _central_sample_policy_rows_ids_heuristic(
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
        heuristic_policy = self._teacher_policy
        if heuristic_policy is None:
            raise RuntimeError("heuristic actor policy backend requires an initialized teacher policy")
        model_started = time.perf_counter()
        if bool(getattr(self, "_actor_behavior_values_required", True)):
            self._central_value_and_advance_actor_rows(
                actors=actors,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
            )
        else:
            if self._should_track_heuristic_actor_hidden_state():
                self._central_advance_actor_rows(
                    actors=actors,
                    obs_steps=obs_steps,
                    actor_steps=actor_steps,
                    row_indices_by_actor=row_indices_by_actor,
                )
            for actor_index, row_indices in enumerate(row_indices_by_actor):
                if row_indices.size:
                    values_outs[actor_index][row_indices] = 0.0
        self._record_batch_timer_ms("central_focal_policy_model", time.perf_counter() - model_started)
        scatter_started = time.perf_counter()
        for actor_index, (actor, batch, obs_step, row_indices) in enumerate(
            zip(actors, batches, obs_steps, row_indices_by_actor, strict=True)
        ):
            if row_indices.size == 0:
                continue
            legal_ids, legal_offsets = _require_ids_offsets(batch)
            legal_action_meta = self._ensure_legal_action_meta(legal_ids, _optional_legal_action_meta(batch))
            chosen_actions = self._heuristic_public_actions_from_ids(
                actor=actor,
                heuristic_policy=heuristic_policy,
                row_indices=row_indices,
                obs_step=obs_step,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                legal_action_meta=legal_action_meta,
            )
            self._maybe_debug_validate_sampled_packed_actions(
                source_label="central:focal:heuristic",
                row_indices=row_indices,
                action_subset=np.asarray(chosen_actions, dtype=np.int64),
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
            )
            write_heuristic_actor_outputs_ids(
                logits_out=None,
                row_indices=row_indices,
                chosen_actions=chosen_actions,
                legal_ids=legal_ids,
                legal_offsets=legal_offsets,
                actions_out=actions_outs[actor_index],
                logp_out=logp_outs[actor_index],
            )
        self._record_batch_timer_ms("central_focal_policy_scatter", time.perf_counter() - scatter_started)

    def _central_sample_policy_rows_ids(
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
        if self._actor_policy_backend != "heuristic_public":
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        heuristic_fraction = self._active_actor_heuristic_fraction()
        if heuristic_fraction >= 1.0:
            self._central_sample_policy_rows_ids_heuristic(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        if heuristic_fraction <= 0.0:
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=row_indices_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
            return
        heuristic_rows_by_actor: list[np.ndarray] = []
        model_rows_by_actor: list[np.ndarray] = []
        any_heuristic_rows = False
        any_model_rows = False
        for actor, row_indices in zip(actors, row_indices_by_actor, strict=True):
            if row_indices.size == 0:
                heuristic_rows_by_actor.append(row_indices)
                model_rows_by_actor.append(row_indices)
                continue
            heuristic_mask = actor.rng.random(row_indices.shape[0]) < heuristic_fraction
            heuristic_rows = row_indices[heuristic_mask]
            model_rows = row_indices[~heuristic_mask]
            heuristic_rows_by_actor.append(heuristic_rows)
            model_rows_by_actor.append(model_rows)
            any_heuristic_rows = any_heuristic_rows or heuristic_rows.size > 0
            any_model_rows = any_model_rows or model_rows.size > 0
        if any_heuristic_rows:
            self._central_sample_policy_rows_ids_heuristic(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=heuristic_rows_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
        if any_model_rows:
            self._central_sample_policy_rows_ids_model(
                actors=actors,
                batches=batches,
                obs_steps=obs_steps,
                actor_steps=actor_steps,
                row_indices_by_actor=model_rows_by_actor,
                values_outs=values_outs,
                actions_outs=actions_outs,
                logp_outs=logp_outs,
            )
