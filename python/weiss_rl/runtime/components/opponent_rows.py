"""Opponent row application helpers for :mod:`weiss_rl.runtime`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID

if TYPE_CHECKING:
    from weiss_rl.runtime.components.actor_state import _ActorState


def _actor_inference_model(actor: _ActorState) -> Any:
    # Resolve lazily through weiss_rl.runtime so tests keep the private wrapper hook.
    from weiss_rl import runtime as runtime_module

    return runtime_module._actor_inference_model(actor)


class QueueRuntimeOpponentRowsMixin:
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def _apply_opponent_rows_mask(
        self,
        *,
        actor: _ActorState,
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
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_mask(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_mask=legal_mask,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                    source_label="opponent:mirror",
                )
                continue
            heuristic_policy = self._heuristic_opponent_policy(policy_id)
            if heuristic_policy is not None:
                if self._should_track_heuristic_actor_hidden_state():
                    self._advance_hidden_only(
                        model=_actor_inference_model(actor),
                        hidden_state=actor.seat_hidden,
                        row_indices=policy_rows,
                        obs_step=obs_step,
                        actor_step=actor_step,
                    )
                chosen_actions = self._heuristic_public_actions_from_mask(
                    actor=actor,
                    heuristic_policy=heuristic_policy,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    legal_mask=legal_mask,
                    profile_name=heuristic_public_profile_name_for_policy_id(policy_id),
                )
                legal_action_ids = [
                    np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(np.uint32, copy=False)
                    for row_index in policy_rows.tolist()
                ]
                self._write_deterministic_logits(
                    logits_out=logits_out,
                    row_indices=policy_rows,
                    chosen_actions=chosen_actions,
                    legal_action_ids=legal_action_ids,
                )
                values_out[policy_rows] = 0.0
                if sample_actions:
                    assert actions_out is not None and logp_out is not None
                    actions_out[policy_rows] = chosen_actions
                    logp_out[policy_rows] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            with self._opponent_model_locks[policy_id]:
                self._apply_policy_rows_mask(
                    model=model,
                    hidden_state=actor.opponent_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_mask=legal_mask,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                    action_selection=str(getattr(self.config, "fixed_model_opponent_action_selection", "sample")),
                    source_label=f"opponent:{policy_id}",
                )

    def _apply_opponent_rows_ids(
        self,
        *,
        actor: _ActorState,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        actor_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
        logits_out: np.ndarray | None,
        values_out: np.ndarray,
        actions_out: np.ndarray | None,
        logp_out: np.ndarray | None,
        rng: np.random.Generator,
        sample_actions: bool = True,
        heuristic_rows_hidden_already_advanced: bool = False,
    ) -> None:
        for policy_id in sorted({str(actor.opponent_policy_id_by_env[index]) for index in row_indices.tolist()}):
            policy_rows = row_indices[actor.opponent_policy_id_by_env[row_indices] == policy_id]
            if not policy_rows.size:
                continue
            if policy_id == MIRROR_OPPONENT_POLICY_ID:
                self._apply_policy_rows_ids(
                    model=_actor_inference_model(actor),
                    hidden_state=actor.seat_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                )
                continue
            heuristic_policy = self._heuristic_opponent_policy(policy_id)
            if heuristic_policy is not None:
                if (
                    not bool(heuristic_rows_hidden_already_advanced)
                    and self._should_track_heuristic_actor_hidden_state()
                ):
                    self._advance_hidden_only(
                        model=_actor_inference_model(actor),
                        hidden_state=actor.seat_hidden,
                        row_indices=policy_rows,
                        obs_step=obs_step,
                        actor_step=actor_step,
                    )
                chosen_actions = self._heuristic_public_actions_from_ids(
                    actor=actor,
                    heuristic_policy=heuristic_policy,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    profile_name=heuristic_public_profile_name_for_policy_id(policy_id),
                )
                self._maybe_debug_validate_sampled_packed_actions(
                    source_label=f"opponent:{policy_id}:heuristic",
                    row_indices=policy_rows,
                    action_subset=np.asarray(chosen_actions, dtype=np.int64),
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                )
                self._write_deterministic_logits_from_packed(
                    logits_out=logits_out,
                    row_indices=policy_rows,
                    chosen_actions=chosen_actions,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                )
                values_out[policy_rows] = 0.0
                if sample_actions:
                    assert actions_out is not None and logp_out is not None
                    actions_out[policy_rows] = chosen_actions
                    logp_out[policy_rows] = 0.0
                continue
            model = self._opponent_models.get(policy_id)
            if model is None:
                raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
            self._advance_hidden_only(
                model=_actor_inference_model(actor),
                hidden_state=actor.seat_hidden,
                row_indices=policy_rows,
                obs_step=obs_step,
                actor_step=actor_step,
            )
            with self._opponent_model_locks[policy_id]:
                self._apply_policy_rows_ids(
                    model=model,
                    hidden_state=actor.opponent_hidden,
                    row_indices=policy_rows,
                    obs_step=obs_step,
                    actor_step=actor_step,
                    legal_ids=legal_ids,
                    legal_offsets=legal_offsets,
                    legal_action_meta=legal_action_meta,
                    logits_out=logits_out,
                    values_out=values_out,
                    actions_out=actions_out,
                    logp_out=logp_out,
                    rng=rng,
                    sample_actions=sample_actions,
                    action_selection=str(getattr(self.config, "fixed_model_opponent_action_selection", "sample")),
                )
