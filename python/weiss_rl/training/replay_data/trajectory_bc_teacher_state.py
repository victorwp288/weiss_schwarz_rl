"""Teacher-guidance state handling for trajectory-BC replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryBcTeacherAuxState:
    teacher_aux_mode: str
    teacher_family_coef: float
    teacher_slot_coef: float
    teacher_hand_coef: float
    teacher_move_source_coef: float
    teacher_attack_type_coef: float
    teacher_action_coef: float
    teacher_same_family_action_coef: float
    teacher_action_margin_coef: float
    teacher_action_margin: float
    teacher_same_family_action_margin_coef: float
    teacher_same_family_action_margin: float
    teacher_exact_action_families: tuple[str, ...]
    teacher_public_heuristic_coef: float
    teacher_public_heuristic_temperature: float
    teacher_public_nonpass_over_pass_coef: float
    teacher_public_nonpass_over_pass_margin: float
    teacher_public_heuristic_families: tuple[str, ...]
    teacher_public_heuristic_profiles: tuple[str, ...]
    teacher_public_heuristic_profile_mode: str
    teacher_public_heuristic_profiles_end_updates: int


def capture_teacher_aux_state(learner: Any) -> TrajectoryBcTeacherAuxState:
    """Capture the learner teacher-guidance knobs mutated during replay BC."""

    return TrajectoryBcTeacherAuxState(
        teacher_aux_mode=str(getattr(learner, "teacher_aux_mode", "always")),
        teacher_family_coef=float(getattr(learner, "teacher_family_coef", 0.0)),
        teacher_slot_coef=float(getattr(learner, "teacher_slot_coef", 0.0)),
        teacher_hand_coef=float(getattr(learner, "teacher_hand_coef", 0.0)),
        teacher_move_source_coef=float(getattr(learner, "teacher_move_source_coef", 0.0)),
        teacher_attack_type_coef=float(getattr(learner, "teacher_attack_type_coef", 0.0)),
        teacher_action_coef=float(getattr(learner, "teacher_action_coef", 0.0)),
        teacher_same_family_action_coef=float(getattr(learner, "teacher_same_family_action_coef", 0.0)),
        teacher_action_margin_coef=float(getattr(learner, "teacher_action_margin_coef", 0.0)),
        teacher_action_margin=float(getattr(learner, "teacher_action_margin", 0.5)),
        teacher_same_family_action_margin_coef=float(getattr(learner, "teacher_same_family_action_margin_coef", 0.0)),
        teacher_same_family_action_margin=float(getattr(learner, "teacher_same_family_action_margin", 0.5)),
        teacher_exact_action_families=tuple(getattr(learner, "teacher_exact_action_families", ())),
        teacher_public_heuristic_coef=float(getattr(learner, "teacher_public_heuristic_coef", 0.0)),
        teacher_public_heuristic_temperature=float(getattr(learner, "teacher_public_heuristic_temperature", 32.0)),
        teacher_public_nonpass_over_pass_coef=float(getattr(learner, "teacher_public_nonpass_over_pass_coef", 0.0)),
        teacher_public_nonpass_over_pass_margin=float(getattr(learner, "teacher_public_nonpass_over_pass_margin", 0.5)),
        teacher_public_heuristic_families=tuple(getattr(learner, "teacher_public_heuristic_families", ())),
        teacher_public_heuristic_profiles=tuple(getattr(learner, "teacher_public_heuristic_profiles", ())),
        teacher_public_heuristic_profile_mode=str(getattr(learner, "teacher_public_heuristic_profile_mode", "")),
        teacher_public_heuristic_profiles_end_updates=int(
            getattr(learner, "teacher_public_heuristic_profiles_end_updates", -1)
        ),
    )


def apply_trajectory_bc_teacher_aux_state(learner: Any, structured_aux: Any) -> None:
    """Apply the temporary teacher-guidance knobs used for replay BC batches."""

    learner.set_teacher_aux_coefs(
        family=float(getattr(structured_aux, "trajectory_bc_teacher_family_coef", 0.05)),
        slot=float(getattr(structured_aux, "trajectory_bc_teacher_slot_coef", 0.05)),
        move_source=float(getattr(structured_aux, "trajectory_bc_teacher_move_source_coef", 0.02)),
        attack_type=float(getattr(structured_aux, "trajectory_bc_teacher_attack_type_coef", 0.02)),
        action=float(getattr(structured_aux, "trajectory_bc_teacher_action_coef", 0.20)),
        same_family_action=float(getattr(structured_aux, "trajectory_bc_teacher_same_family_action_coef", 0.60)),
        action_margin=0.0,
        same_family_action_margin=float(
            getattr(structured_aux, "trajectory_bc_teacher_same_family_action_margin_coef", 0.10)
        ),
        same_family_action_margin_value=float(
            getattr(structured_aux, "trajectory_bc_teacher_same_family_action_margin", 0.5)
        ),
        public_heuristic=0.0,
        public_nonpass_over_pass=0.0,
        exact_action_families=(),
    )
    learner.teacher_aux_mode = "warmstart_only"


def restore_teacher_aux_state(learner: Any, state: TrajectoryBcTeacherAuxState) -> None:
    """Restore a learner teacher-guidance snapshot captured before replay BC."""

    learner.set_teacher_aux_coefs(
        family=state.teacher_family_coef,
        slot=state.teacher_slot_coef,
        hand=state.teacher_hand_coef,
        move_source=state.teacher_move_source_coef,
        attack_type=state.teacher_attack_type_coef,
        action=state.teacher_action_coef,
        same_family_action=state.teacher_same_family_action_coef,
        action_margin=state.teacher_action_margin_coef,
        action_margin_value=state.teacher_action_margin,
        same_family_action_margin=state.teacher_same_family_action_margin_coef,
        same_family_action_margin_value=state.teacher_same_family_action_margin,
        exact_action_families=state.teacher_exact_action_families,
        public_heuristic=state.teacher_public_heuristic_coef,
        public_heuristic_temperature=state.teacher_public_heuristic_temperature,
        public_nonpass_over_pass=state.teacher_public_nonpass_over_pass_coef,
        public_nonpass_over_pass_margin=state.teacher_public_nonpass_over_pass_margin,
        public_heuristic_families=state.teacher_public_heuristic_families,
        public_heuristic_profiles=state.teacher_public_heuristic_profiles,
        public_heuristic_profile_mode=state.teacher_public_heuristic_profile_mode,
        public_heuristic_profiles_end_updates=state.teacher_public_heuristic_profiles_end_updates,
    )
    learner.teacher_aux_mode = state.teacher_aux_mode


__all__ = [
    "TrajectoryBcTeacherAuxState",
    "apply_trajectory_bc_teacher_aux_state",
    "capture_teacher_aux_state",
    "restore_teacher_aux_state",
]
