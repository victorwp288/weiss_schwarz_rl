"""QueueRuntime adapters for heuristic fast paths and teacher labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.runtime.components.heuristic_fast_path import (
    actor_fixed_opponents_all_heuristic_public,
    can_collect_all_heuristic_ids_fast,
    can_collect_all_heuristic_ids_native_rollout,
    simulator_native_fixed_opponent_available,
)
from weiss_rl.runtime.components.teacher_labels import (
    selected_teacher_label_profile,
    teacher_guidance_active_for_collection,
    teacher_label_arrays,
    teacher_labels_from_actions,
    teacher_labels_from_ids,
    teacher_labels_from_mask,
)

TeacherLabelArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class QueueRuntimeTeacherHeuristicMixin:
    """Bridge QueueRuntime state into pure heuristic and teacher-label helpers."""

    def _simulator_native_fixed_opponent_available(self: Any, actor: Any | None) -> bool:
        return simulator_native_fixed_opponent_available(
            actor,
            fixed_opponent_backend=str(getattr(self, "_fixed_opponent_backend", "python_batched")),
        )

    def _actor_fixed_opponents_all_heuristic_public(self: Any, actor: Any) -> bool:
        return actor_fixed_opponents_all_heuristic_public(
            actor,
            fixed_opponent_policy_is_active=self._fixed_opponent_policy_is_active,
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        )

    def _can_collect_all_heuristic_ids_fast(self: Any, actor: Any) -> bool:
        return can_collect_all_heuristic_ids_fast(
            actor,
            actor_policy_backend=str(getattr(self, "_actor_policy_backend", "model")),
            active_actor_heuristic_fraction=self._active_actor_heuristic_fraction(),
            fixed_opponent_backend=str(getattr(self, "_fixed_opponent_backend", "python_batched")),
            teacher_policy=self._teacher_policy,
            league_config=getattr(self, "_league_config", None),
            active_heuristic_public_mix_fraction=self._active_heuristic_public_mix_fraction(),
            fixed_opponent_policy_is_active=self._fixed_opponent_policy_is_active,
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        )

    def _can_collect_all_heuristic_ids_native_rollout(self: Any, actor: Any) -> bool:
        config = getattr(self, "config", None)
        if (
            bool(getattr(config, "mulligan_force_confirm_after_select", False))
            or bool(getattr(config, "force_pass_over_main_move_only", False))
            or bool(getattr(config, "force_attack_over_pass_when_attack_legal", False))
        ):
            return False
        return can_collect_all_heuristic_ids_native_rollout(
            actor,
            heuristic_native_rollout_enabled=bool(getattr(self, "_heuristic_native_rollout_enabled", False)),
            actor_policy_backend=str(getattr(self, "_actor_policy_backend", "model")),
            active_actor_heuristic_fraction=self._active_actor_heuristic_fraction(),
            fixed_opponent_backend=str(getattr(self, "_fixed_opponent_backend", "python_batched")),
            teacher_policy=self._teacher_policy,
            league_config=getattr(self, "_league_config", None),
            active_heuristic_public_mix_fraction=self._active_heuristic_public_mix_fraction(),
            fixed_opponent_policy_is_active=self._fixed_opponent_policy_is_active,
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            actor_behavior_values_required=bool(getattr(self, "_actor_behavior_values_required", True)),
            should_track_heuristic_actor_hidden_state=self._should_track_heuristic_actor_hidden_state(),
        )

    def _teacher_guidance_active_for_collection(self: Any) -> bool:
        return teacher_guidance_active_for_collection(
            enabled=bool(getattr(self, "_teacher_guidance_enabled", False)),
            teacher_aux_mode=str(getattr(self, "_teacher_aux_mode", "always")),
            warmstart_updates=int(getattr(self, "_teacher_guidance_warmstart_updates", 0)),
            current_learner_update=int(getattr(self, "_current_learner_update", 0)),
        )

    def _teacher_label_arrays(self: Any, num_rows: int) -> TeacherLabelArrays:
        return teacher_label_arrays(num_rows)

    def _teacher_label_policy_for_current_update(self: Any) -> Any | None:
        policy_by_profile = getattr(self, "_teacher_policy_by_profile", None)
        if not isinstance(policy_by_profile, dict) or not policy_by_profile:
            return getattr(self, "_teacher_policy", None)
        profile = selected_teacher_label_profile(
            getattr(self, "_teacher_label_profiles", ("base",)),
            profile_mode=str(getattr(self, "_teacher_label_profile_mode", "mixture")),
            update_count=int(getattr(self, "_current_learner_update", 0)),
            end_updates=int(getattr(self, "_teacher_label_profiles_end_updates", -1)),
        )
        return policy_by_profile.get(profile) or policy_by_profile.get("base") or getattr(self, "_teacher_policy", None)

    def _teacher_labels_from_actions(
        self: Any,
        *,
        row_indices: np.ndarray,
        chosen_actions: np.ndarray,
        num_rows: int,
    ) -> TeacherLabelArrays:
        return teacher_labels_from_actions(
            row_indices=row_indices,
            chosen_actions=chosen_actions,
            num_rows=int(num_rows),
            guidance_active=self._teacher_guidance_active_for_collection(),
            action_catalog=self._teacher_action_catalog,
            family_index=self._teacher_family_index,
            attack_type_index=self._teacher_attack_type_index,
        )

    def _teacher_labels_from_ids(
        self: Any,
        *,
        focal_rows: np.ndarray,
        decision_kind: np.ndarray,
        obs_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None = None,
        counters: dict[str, int] | None = None,
    ) -> TeacherLabelArrays:
        return teacher_labels_from_ids(
            focal_rows=focal_rows,
            decision_kind=decision_kind,
            obs_step=obs_step,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=legal_action_meta,
            counters=counters,
            guidance_active=self._teacher_guidance_active_for_collection(),
            teacher_policy=self._teacher_label_policy_for_current_update(),
            action_catalog=self._teacher_action_catalog,
            family_index=self._teacher_family_index,
            attack_type_index=self._teacher_attack_type_index,
            select_actions_from_ids=self._heuristic_public_actions_from_ids,
        )

    def _teacher_labels_from_mask(
        self: Any,
        *,
        focal_rows: np.ndarray,
        decision_kind: np.ndarray,
        obs_step: np.ndarray,
        legal_mask: np.ndarray,
        counters: dict[str, int] | None = None,
    ) -> TeacherLabelArrays:
        return teacher_labels_from_mask(
            focal_rows=focal_rows,
            decision_kind=decision_kind,
            obs_step=obs_step,
            legal_mask=legal_mask,
            counters=counters,
            guidance_active=self._teacher_guidance_active_for_collection(),
            teacher_policy=self._teacher_label_policy_for_current_update(),
            action_catalog=self._teacher_action_catalog,
            family_index=self._teacher_family_index,
            attack_type_index=self._teacher_attack_type_index,
            select_actions_from_mask=self._heuristic_public_actions_from_mask,
        )
