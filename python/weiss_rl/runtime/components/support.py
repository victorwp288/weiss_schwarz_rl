"""Support-method mixin for :class:`weiss_rl.runtime.QueueRuntime`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime.components.action_surface import (
    filter_batch_main_move_only_rows_to_pass,
    filter_batch_mulligan_select_after_select,
    filter_batch_pass_when_attack_available,
)
from weiss_rl.runtime.components.actor_routing import policy_train_mask_for_actor, split_focal_actor_rows
from weiss_rl.runtime.components.actor_state import _ActorState
from weiss_rl.runtime.components.batching.impala_learner_batch import build_impala_learner_batch
from weiss_rl.runtime.components.batching.ppo_learner_batch import build_ppo_learner_batch
from weiss_rl.runtime.components.bootstrap import bootstrap_values_for_unroll
from weiss_rl.runtime.components.metrics import build_runtime_metrics
from weiss_rl.runtime.components.opponent_context import initial_seat_hidden_for_opponents
from weiss_rl.runtime.components.outcomes import update_outcomes, update_outcomes_from_transition_arrays
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID
from weiss_rl.runtime.components.policy_inference.deterministic_logits import (
    write_deterministic_logits,
    write_deterministic_logits_from_packed,
)
from weiss_rl.runtime.components.types import PendingUnroll, RuntimeUnroll


class QueueRuntimeSupportMixin:
    def _filter_action_surface_for_batch(
        self: Any,
        batch: DecisionBoundaryBatch,
        *,
        counters: dict[str, int] | None = None,
        action_sequence_state: Any | None = None,
    ) -> DecisionBoundaryBatch:
        if batch.ids_offsets is None:
            return batch
        legal_action_meta = getattr(batch, "legal_action_meta", None)
        if legal_action_meta is None:
            return batch
        filtered_batch = batch
        family_index = getattr(self, "_action_family_index", {})
        if bool(getattr(self.config, "mulligan_force_confirm_after_select", False)):
            last_action_arg0_index = int(getattr(self, "_last_action_arg0_obs_index", -1))
            filtered_batch, result = filter_batch_mulligan_select_after_select(
                filtered_batch,
                last_action_arg0_index=last_action_arg0_index,
                mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
                mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
            )
            if result.filtered_actions > 0 and counters is not None:
                counters["mulligan_force_confirm_after_select_rows"] += int(result.filtered_rows)
                counters["mulligan_force_confirm_after_select_actions"] += int(result.filtered_actions)
        if bool(getattr(self.config, "force_pass_over_main_move_only", False)):
            allow_main_move_only_rows = None
            max_consecutive_main_moves = int(getattr(self.config, "main_move_only_max_consecutive", 0))
            if max_consecutive_main_moves > 0 and action_sequence_state is not None:
                consecutive = np.asarray(action_sequence_state.consecutive_main_moves_by_env, dtype=np.int32)
                if consecutive.shape == (int(batch.obs.shape[0]),):
                    allow_main_move_only_rows = consecutive < max_consecutive_main_moves
            filtered_batch, result = filter_batch_main_move_only_rows_to_pass(
                filtered_batch,
                pass_action_id=int(self.config.pass_action_id),
                main_move_family_id=int(family_index.get("main_move", -1)),
                allow_main_move_only_rows=allow_main_move_only_rows,
            )
            if result.filtered_actions > 0 and counters is not None:
                counters["main_move_only_force_pass_rows"] += int(result.filtered_rows)
                counters["main_move_only_force_pass_actions"] += int(result.filtered_actions)
        if bool(getattr(self.config, "force_attack_over_pass_when_attack_legal", False)):
            filtered_batch, result = filter_batch_pass_when_attack_available(
                filtered_batch,
                pass_action_id=int(self.config.pass_action_id),
                attack_family_id=int(family_index.get("attack", -1)),
            )
            if result.filtered_actions > 0 and counters is not None:
                counters["attack_available_force_attack_rows"] += int(result.filtered_rows)
                counters["attack_available_force_attack_actions"] += int(result.filtered_actions)
        return filtered_batch

    def _write_deterministic_logits(
        self: Any,
        *,
        logits_out: np.ndarray | None,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        legal_action_ids: Sequence[np.ndarray],
    ) -> None:
        write_deterministic_logits(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_action_ids=legal_action_ids,
            action_dim=int(self.action_dim),
        )

    def _write_deterministic_logits_from_packed(
        self: Any,
        *,
        logits_out: np.ndarray | None,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
    ) -> None:
        write_deterministic_logits_from_packed(
            logits_out=logits_out,
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )

    def _update_outcomes(
        self: Any,
        *,
        actor: _ActorState,
        acting_seat: np.ndarray,
        terminal_batch: DecisionBoundaryBatch,
        done: np.ndarray,
        counters: dict[str, int] | None = None,
    ) -> None:
        update_outcomes(
            outcome_tracker=self._outcomes,
            opponent_policy_id_by_env=actor.opponent_policy_id_by_env,
            focal_seat_by_env=actor.focal_seat_by_env,
            acting_seat=acting_seat,
            terminal_batch=terminal_batch,
            done=done,
            mirror_policy_id=MIRROR_OPPONENT_POLICY_ID,
            counters=counters,
        )

    def _update_outcomes_from_transition_arrays(
        self: Any,
        *,
        actor: _ActorState,
        acting_seat: np.ndarray,
        rewards: np.ndarray,
        truncated: np.ndarray,
        done: np.ndarray,
        counters: dict[str, int] | None = None,
    ) -> None:
        update_outcomes_from_transition_arrays(
            outcome_tracker=self._outcomes,
            opponent_policy_id_by_env=actor.opponent_policy_id_by_env,
            focal_seat_by_env=actor.focal_seat_by_env,
            acting_seat=acting_seat,
            rewards=rewards,
            truncated=truncated,
            done=done,
            mirror_policy_id=MIRROR_OPPONENT_POLICY_ID,
            counters=counters,
        )

    def _split_focal_actor_rows(
        self: Any,
        *,
        actor: _ActorState,
        focal_indices: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._actor_policy_backend != "heuristic_public":
            return focal_indices, np.zeros((0,), dtype=np.int64)
        return split_focal_actor_rows(
            focal_indices=focal_indices,
            rng=rng,
            teacher_policy_available=self._teacher_policy is not None,
            force_model_policy_lane=bool(getattr(actor, "force_model_policy_lane", False)),
            heuristic_fraction=float(self._active_actor_heuristic_fraction()),
        )

    def _policy_train_mask_for_actor(
        self: Any,
        *,
        actor: _ActorState,
        focal_rows: np.ndarray,
        include_mirror_opponent_rows: bool = True,
    ) -> np.ndarray:
        return policy_train_mask_for_actor(
            focal_rows=focal_rows,
            train_on_heuristic_actor_rows=bool(getattr(self, "_train_on_heuristic_actor_rows", True)),
            actor_policy_backend=str(self._actor_policy_backend),
            force_model_policy_lane=bool(getattr(actor, "force_model_policy_lane", False)),
            heuristic_fraction=float(self._active_actor_heuristic_fraction()),
            opponent_policy_id_by_env=(
                getattr(actor, "opponent_policy_id_by_env", None) if bool(include_mirror_opponent_rows) else None
            ),
        )

    def _trajectory_retention_mask_for_actor(
        self: Any,
        *,
        actor: _ActorState,
        focal_rows: np.ndarray,
    ) -> np.ndarray | None:
        if not bool(getattr(self, "_trajectory_retention_enabled", False)):
            return None
        focal = np.asarray(focal_rows, dtype=np.bool_)
        opponent_ids = np.asarray(getattr(actor, "opponent_policy_id_by_env", ()), dtype=object)
        if opponent_ids.shape != focal.shape:
            raise ValueError(f"opponent_policy_id_by_env must have shape {focal.shape}, got {opponent_ids.shape}")

        retained_ids: set[str] = set(getattr(self, "_trajectory_retention_policy_ids", ()))
        sources = set(getattr(self, "_trajectory_retention_sources", ()))
        if "champions" in sources:
            retained_ids.update(str(policy_id) for policy_id in getattr(self, "_opponent_champion_ids", ()))
        if "recent" in sources:
            retained_ids.update(str(policy_id) for policy_id in getattr(self, "_opponent_recent_ids", ()))
        if "hard_negatives" in sources:
            retained_ids.update(str(policy_id) for policy_id in getattr(self, "_opponent_hard_negative_ids", ()))
        if "warmup_snapshots" in sources and self._active_warmup_snapshot_mix_fraction() > 0.0:
            retained_ids.update(str(policy_id) for policy_id in getattr(self, "_opponent_candidate_ids", ()))
        if "all_model" in sources:
            retained_ids.update(str(policy_id) for policy_id in getattr(self, "_opponent_models", {}).keys())

        model_ids = {str(policy_id) for policy_id in getattr(self, "_opponent_models", {}).keys()}
        retained_ids = {policy_id for policy_id in retained_ids if policy_id in model_ids}
        if not retained_ids:
            return np.zeros(focal.shape, dtype=np.bool_)
        return np.logical_and(~focal, np.isin(opponent_ids, tuple(retained_ids))).astype(np.bool_, copy=False)

    def _build_learner_batch(
        self: Any,
        unrolls: Sequence[PendingUnroll],
        *,
        gamma: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> dict[str, Any]:
        runtime_config = getattr(self, "config", None)
        return build_impala_learner_batch(
            unrolls,
            action_dim=int(self.action_dim),
            gamma=gamma,
            truncation_reward=truncation_reward,
            truncation_bootstrap_value=truncation_bootstrap_value,
            vtrace_rho_bar=vtrace_rho_bar,
            vtrace_c_bar=vtrace_c_bar,
            terminal_outcome_backfill_reward=float(getattr(runtime_config, "terminal_outcome_backfill_reward", 0.0)),
            terminal_outcome_trace_backfill_reward=float(
                getattr(runtime_config, "terminal_outcome_trace_backfill_reward", 0.0)
            ),
            record_batch_timer_ms=self._record_batch_timer_ms,
        )

    def _build_ppo_batch(
        self: Any,
        unrolls: Sequence[PendingUnroll],
        *,
        gamma: float,
        gae_lambda: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
    ) -> dict[str, Any]:
        return build_ppo_learner_batch(
            unrolls,
            action_dim=int(self.action_dim),
            gamma=gamma,
            gae_lambda=gae_lambda,
            truncation_reward=truncation_reward,
            truncation_bootstrap_value=truncation_bootstrap_value,
            record_batch_timer_ms=self._record_batch_timer_ms,
        )

    def _bootstrap_values(self: Any, unroll: RuntimeUnroll) -> np.ndarray:
        bootstrap_device = self._device
        if self._bootstrap_models is not None:
            actor_model = self._bootstrap_models[int(unroll.actor_id)]
            bootstrap_device = self._bootstrap_model_devices[int(unroll.actor_id)]
        else:
            actor_model = self._actors[int(unroll.actor_id)].model
        return bootstrap_values_for_unroll(
            unroll=unroll,
            actor_model=actor_model,
            bootstrap_device=bootstrap_device,
            actor_amp_enabled=bool(self._actor_amp_enabled),
        )

    def _runtime_metrics(
        self: Any,
        selected: Sequence[PendingUnroll],
        *,
        occupancy_samples: Sequence[float],
    ) -> dict[str, float]:
        now = time.time()
        metrics, next_cumulative_env_steps = build_runtime_metrics(
            selected=selected,
            occupancy_samples=occupancy_samples,
            now=now,
            runtime_start=self._runtime_start,
            runtime_last_metrics_time=self._runtime_last_metrics_time,
            runtime_cumulative_env_steps=self._runtime_cumulative_env_steps,
            last_published_snapshot_version=self._last_published_snapshot_version,
            current_learner_update=int(getattr(self, "_current_learner_update", 0)),
            effective_learner_update=int(getattr(self, "_effective_learner_update", 0)),
            actor_heuristic_fraction_active=float(self._active_actor_heuristic_fraction()),
            mirror_mix_fraction_active=float(self._active_mirror_mix_fraction()),
            heuristic_public_mix_fraction_active=float(self._active_heuristic_public_mix_fraction()),
            heuristic_public_variant_mix_fraction_active=float(self._active_heuristic_public_variant_mix_fraction()),
            warmup_snapshot_mix_fraction_active=float(self._active_warmup_snapshot_mix_fraction()),
            pfsp_pool_size=int(self._pfsp_pool_size),
            pfsp_quarantined_opponents=int(self._pfsp_quarantined_opponents),
            pfsp_champion_pool_size=int(self._pfsp_champion_pool_size),
            pfsp_recent_pool_size=int(self._pfsp_recent_pool_size),
            pfsp_hard_negative_pool_size=int(self._pfsp_hard_negative_pool_size),
            pfsp_last_sampled_envs=int(self._pfsp_last_sampled_envs),
            pfsp_last_mirror_envs=int(self._pfsp_last_mirror_envs),
            pfsp_last_heuristic_public_envs=int(self._pfsp_last_heuristic_public_envs),
            pfsp_last_heuristic_public_variant_envs=int(getattr(self, "_pfsp_last_heuristic_public_variant_envs", 0)),
            pfsp_last_noleague_baseline_envs=int(getattr(self, "_pfsp_last_noleague_baseline_envs", 0)),
            pfsp_last_champion_envs=int(self._pfsp_last_champion_envs),
            pfsp_last_recent_envs=int(self._pfsp_last_recent_envs),
            pfsp_last_hard_negative_envs=int(self._pfsp_last_hard_negative_envs),
            pfsp_last_warmup_snapshot_envs=int(getattr(self, "_pfsp_last_warmup_snapshot_envs", 0)),
            pfsp_epoch=int(self._pfsp_epoch),
        )
        self._runtime_last_metrics_time = now
        self._runtime_cumulative_env_steps = next_cumulative_env_steps
        return metrics

    def _reset_done_rows(self: Any, actor: _ActorState, done: np.ndarray) -> DecisionBoundaryBatch:
        try:
            return actor.env.reset_done(done)
        except RuntimeError:
            full_reset = np.ones(actor.focal_seat_by_env.shape, dtype=np.bool_)
            self._assign_episode_roles(actor, full_reset, initial=True)
            initial_hidden = initial_seat_hidden_for_opponents(
                actor.model,
                int(self.config.envs_per_actor),
                device=self._device,
                opponent_policy_ids=cast(Sequence[object], actor.opponent_policy_id_by_env),
            ).clone()
            actor.seat_hidden = initial_hidden.clone()
            actor.opponent_hidden = initial_seat_hidden_for_opponents(
                actor.model,
                int(self.config.envs_per_actor),
                device=self._device,
            )
            fallback_seed = int(actor.rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            return actor.env.reset(seed=fallback_seed)

    def _reset_actor_state_for_fixed_opponents(self: Any, actor: _ActorState) -> None:
        full_reset = np.ones(actor.focal_seat_by_env.shape, dtype=np.bool_)
        self._assign_episode_roles(actor, full_reset, initial=True)
        initial_hidden = initial_seat_hidden_for_opponents(
            actor.model,
            int(self.config.envs_per_actor),
            device=self._device,
            opponent_policy_ids=cast(Sequence[object], actor.opponent_policy_id_by_env),
        ).clone()
        actor.seat_hidden = initial_hidden.clone()
        actor.opponent_hidden = initial_seat_hidden_for_opponents(
            actor.model,
            int(self.config.envs_per_actor),
            device=self._device,
        )
        actor.current_batch = self._reset_done_rows(actor, full_reset)
