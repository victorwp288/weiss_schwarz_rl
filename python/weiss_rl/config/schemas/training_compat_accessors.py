"""Legacy flat accessors for nested training config records."""

from __future__ import annotations


class TrainingConfigCompatibilityMixin:
    """Expose historical flat field names while the config stays grouped."""

    @property
    def unroll_length(self) -> int:
        return int(self.rollout.unroll_length)

    @property
    def batch_unrolls_per_update(self) -> int:
        return int(self.rollout.batch_unrolls_per_update)

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.learning_rate)

    @property
    def grad_norm_clip(self) -> float:
        return float(self.optimizer.grad_norm_clip)

    @property
    def value_loss_coef(self) -> float:
        return float(self.optimizer.value_loss_coef)

    @property
    def entropy_coef(self) -> float:
        return float(self.exploration.entropy_coef)

    @property
    def entropy_anneal_to(self) -> float:
        return float(self.exploration.entropy_anneal_to)

    @property
    def entropy_anneal_steps_updates(self) -> int:
        return int(self.exploration.entropy_anneal_steps_updates)

    @property
    def entropy_scope(self) -> str:
        return str(self.exploration.entropy_scope)

    @property
    def actor_sampling_temperature(self) -> float:
        return float(self.exploration.actor_sampling_temperature)

    @property
    def mixed_precision(self) -> bool:
        return bool(self.precision.mixed_precision)

    @property
    def compile_learner(self) -> bool:
        return bool(self.precision.compile_learner)

    @property
    def compile_actor_inference(self) -> bool:
        return bool(self.precision.compile_actor_inference)

    @property
    def masking_math_float32(self) -> bool:
        return bool(self.precision.masking_math_float32)

    @property
    def checkpoint_interval_updates(self) -> int:
        return int(self.checkpointing.checkpoint_interval_updates)

    @property
    def snapshot_interval_updates(self) -> int:
        return int(self.checkpointing.snapshot_interval_updates)

    @property
    def actor_reload_interval_updates(self) -> int:
        return int(self.checkpointing.actor_reload_interval_updates)

    @property
    def vtrace_rho_bar(self) -> float:
        return float(self.vtrace.rho_bar)

    @property
    def vtrace_c_bar(self) -> float:
        return float(self.vtrace.c_bar)

    @property
    def ppo_clip_epsilon(self) -> float:
        return float(self.ppo.clip_epsilon)

    @property
    def ppo_value_clip_epsilon(self) -> float:
        return float(self.ppo.value_clip_epsilon)

    @property
    def ppo_gae_lambda(self) -> float:
        return float(self.ppo.gae_lambda)

    @property
    def ppo_epochs(self) -> int:
        return int(self.ppo.epochs)

    @property
    def ppo_target_kl(self) -> float:
        return float(self.ppo.target_kl)

    @property
    def ppo_normalize_advantages(self) -> bool:
        return bool(self.ppo.normalize_advantages)

    @property
    def structured_aux_enabled(self) -> bool:
        return bool(self.structured_aux.enabled)

    @property
    def structured_metrics_mode(self) -> str:
        return self.structured_metrics.mode

    @property
    def teacher_aux_mode(self) -> str:
        return self.teacher_aux.mode

    @property
    def mulligan_force_confirm_after_select(self) -> bool:
        return bool(self.action_surface.mulligan_force_confirm_after_select)

    @property
    def force_pass_over_main_move_only(self) -> bool:
        return bool(self.action_surface.force_pass_over_main_move_only)

    @property
    def main_move_only_max_consecutive(self) -> int:
        return int(self.action_surface.main_move_only_max_consecutive)

    @property
    def force_attack_over_pass_when_attack_legal(self) -> bool:
        return bool(self.action_surface.force_attack_over_pass_when_attack_legal)

    @property
    def teacher_family_coef(self) -> float:
        return float(self.structured_aux.teacher_family_coef)

    @property
    def teacher_slot_coef(self) -> float:
        return float(self.structured_aux.teacher_slot_coef)

    @property
    def teacher_hand_coef(self) -> float:
        return float(self.structured_aux.teacher_hand_coef)

    @property
    def teacher_move_source_coef(self) -> float:
        return float(self.structured_aux.teacher_move_source_coef)

    @property
    def teacher_attack_type_coef(self) -> float:
        return float(self.structured_aux.teacher_attack_type_coef)

    @property
    def teacher_action_coef(self) -> float:
        return float(self.structured_aux.teacher_action_coef)

    @property
    def teacher_same_family_action_coef(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_coef)

    @property
    def teacher_action_margin_coef(self) -> float:
        return float(self.structured_aux.teacher_action_margin_coef)

    @property
    def teacher_action_margin(self) -> float:
        return float(self.structured_aux.teacher_action_margin)

    @property
    def teacher_same_family_action_margin_coef(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_margin_coef)

    @property
    def teacher_same_family_action_margin(self) -> float:
        return float(self.structured_aux.teacher_same_family_action_margin)

    @property
    def teacher_supervised_start_updates(self) -> int:
        return int(self.structured_aux.teacher_supervised_start_updates)

    @property
    def teacher_supervised_end_updates(self) -> int:
        return int(self.structured_aux.teacher_supervised_end_updates)

    @property
    def teacher_supervised_final_scale(self) -> float:
        return float(self.structured_aux.teacher_supervised_final_scale)

    @property
    def teacher_exact_action_families(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_exact_action_families)

    @property
    def teacher_public_heuristic_coef(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_coef)

    @property
    def teacher_public_heuristic_start_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_start_updates)

    @property
    def teacher_public_heuristic_end_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_end_updates)

    @property
    def teacher_public_heuristic_final_coef(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_final_coef)

    @property
    def teacher_public_heuristic_temperature(self) -> float:
        return float(self.structured_aux.teacher_public_heuristic_temperature)

    @property
    def teacher_public_nonpass_over_pass_coef(self) -> float:
        return float(self.structured_aux.teacher_public_nonpass_over_pass_coef)

    @property
    def teacher_public_nonpass_over_pass_margin(self) -> float:
        return float(self.structured_aux.teacher_public_nonpass_over_pass_margin)

    @property
    def teacher_public_heuristic_families(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_public_heuristic_families)

    @property
    def teacher_public_heuristic_profiles(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.teacher_public_heuristic_profiles)

    @property
    def teacher_public_heuristic_profile_mode(self) -> str:
        return str(self.structured_aux.teacher_public_heuristic_profile_mode)

    @property
    def teacher_public_heuristic_profiles_end_updates(self) -> int:
        return int(self.structured_aux.teacher_public_heuristic_profiles_end_updates)

    @property
    def policy_anchor_coef(self) -> float:
        return float(self.structured_aux.policy_anchor_coef)

    @property
    def policy_anchor_top_action_coef(self) -> float:
        return float(self.structured_aux.policy_anchor_top_action_coef)

    @property
    def policy_anchor_temperature(self) -> float:
        return float(self.structured_aux.policy_anchor_temperature)

    @property
    def trajectory_retention_coef(self) -> float:
        return float(self.structured_aux.trajectory_retention_coef)

    @property
    def trajectory_retention_policy_ids(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.trajectory_retention_policy_ids)

    @property
    def trajectory_retention_sources(self) -> tuple[str, ...]:
        return tuple(self.structured_aux.trajectory_retention_sources)

    @property
    def trajectory_bc_dataset_path(self) -> str:
        return str(self.structured_aux.trajectory_bc_dataset_path)

    @property
    def trajectory_bc_enabled(self) -> bool:
        return bool(str(self.structured_aux.trajectory_bc_dataset_path).strip()) and (
            int(self.structured_aux.trajectory_bc_every_updates) > 0
        )

    @property
    def paired_swing_enabled(self) -> bool:
        return bool(str(self.structured_aux.paired_swing_dataset_path).strip()) and (
            int(self.structured_aux.paired_swing_every_updates) > 0
        )

    @property
    def paired_outcome_preference_enabled(self) -> bool:
        return bool(str(self.structured_aux.paired_outcome_preference_dataset_path).strip()) and (
            int(self.structured_aux.paired_outcome_preference_every_updates) > 0
        )

    @property
    def structured_warmstart_enabled(self) -> bool:
        return bool(self.structured_warmstart.enabled) and int(self.structured_warmstart.updates) > 0


__all__ = ["TrainingConfigCompatibilityMixin"]
