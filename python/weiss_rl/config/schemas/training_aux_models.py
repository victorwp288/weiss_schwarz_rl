"""Structured auxiliary training config records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TrainingTrajectoryBcFocusGroupConfig:
    name: str
    source_labels: tuple[str, ...]
    fraction: float


@dataclass(frozen=True, slots=True)
class TrainingStructuredAuxConfig:
    enabled: bool = False
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_action_margin_coef: float = 0.0
    teacher_action_margin: float = 0.5
    teacher_same_family_action_margin_coef: float = 0.0
    teacher_same_family_action_margin: float = 0.5
    teacher_supervised_start_updates: int = 0
    teacher_supervised_end_updates: int = -1
    teacher_supervised_final_scale: float = 1.0
    teacher_exact_action_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_start_updates: int = 0
    teacher_public_heuristic_end_updates: int = -1
    teacher_public_heuristic_final_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_nonpass_over_pass_coef: float = 0.0
    teacher_public_nonpass_over_pass_margin: float = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1
    policy_anchor_coef: float = 0.0
    policy_anchor_top_action_coef: float = 0.0
    policy_anchor_temperature: float = 1.0
    trajectory_retention_coef: float = 0.0
    trajectory_retention_policy_ids: tuple[str, ...] = field(default_factory=tuple)
    trajectory_retention_sources: tuple[str, ...] = field(default_factory=lambda: ("champions",))
    trajectory_bc_dataset_path: str = ""
    trajectory_bc_every_updates: int = 0
    trajectory_bc_aux_updates: int = 1
    trajectory_bc_batch_episodes: int = 8
    trajectory_bc_seed: int = 20260516
    trajectory_bc_focus_source_labels: tuple[str, ...] = field(default_factory=tuple)
    trajectory_bc_focus_fraction: float = 0.0
    trajectory_bc_focus_groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...] = field(default_factory=tuple)
    trajectory_bc_teacher_family_coef: float = 0.05
    trajectory_bc_teacher_slot_coef: float = 0.05
    trajectory_bc_teacher_move_source_coef: float = 0.02
    trajectory_bc_teacher_attack_type_coef: float = 0.02
    trajectory_bc_teacher_action_coef: float = 0.20
    trajectory_bc_teacher_same_family_action_coef: float = 0.60
    trajectory_bc_teacher_same_family_action_margin_coef: float = 0.10
    trajectory_bc_teacher_same_family_action_margin: float = 0.5
    paired_swing_dataset_path: str = ""
    paired_swing_every_updates: int = 0
    paired_swing_aux_updates: int = 1
    paired_swing_batch_episodes: int = 8
    paired_swing_seed: int = 20260519
    paired_swing_focus_source_labels: tuple[str, ...] = field(default_factory=tuple)
    paired_swing_focus_fraction: float = 0.0
    paired_swing_focus_groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...] = field(default_factory=tuple)
    paired_swing_margin: float = 0.35
    paired_swing_coef: float = 0.08
    paired_swing_positive_action_source: str = "teacher_action"
    paired_swing_negative_action_source: str = "actions"
    paired_swing_conflict_filter: str = "none"
    paired_swing_loss_scope: str = "row"
    paired_swing_compare_to: str = "negative"
    paired_outcome_preference_dataset_path: str = ""
    paired_outcome_preference_every_updates: int = 0
    paired_outcome_preference_aux_updates: int = 1
    paired_outcome_preference_batch_episodes: int = 8
    paired_outcome_preference_seed: int = 20260520
    paired_outcome_preference_coef: float = 0.05
    paired_outcome_preference_beta: float = 0.1
    paired_outcome_preference_aggregation: str = "mean"
    paired_outcome_preference_group_balance: bool = False


@dataclass(frozen=True, slots=True)
class TrainingStructuredWarmstartConfig:
    enabled: bool = False
    updates: int = 0
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_hand_coef: float = 0.0
    teacher_move_source_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    teacher_action_coef: float = 0.0
    teacher_same_family_action_coef: float = 0.0
    teacher_public_heuristic_coef: float = 0.0
    teacher_public_heuristic_temperature: float = 32.0
    teacher_public_heuristic_families: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profiles: tuple[str, ...] = field(default_factory=tuple)
    teacher_public_heuristic_profile_mode: str = "mixture"
    teacher_public_heuristic_profiles_end_updates: int = -1


__all__ = [
    "TrainingStructuredAuxConfig",
    "TrainingStructuredWarmstartConfig",
    "TrainingTrajectoryBcFocusGroupConfig",
]
