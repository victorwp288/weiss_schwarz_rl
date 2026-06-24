"""Teacher-guidance section for learning-progress diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.diagnostics.progress.learning_progress_math import _fraction_values, _numeric_values, _window_summary


@dataclass(frozen=True, slots=True)
class TeacherGuidanceSeries:
    teacher_public_heuristic_coef_active_values: list[float]
    teacher_hand_coef_active_values: list[float]
    teacher_aux_loss_values: list[float]
    teacher_main_play_slot_accuracy_values: list[float]
    teacher_hand_accuracy_values: list[float]
    teacher_main_play_hand_accuracy_values: list[float]
    teacher_clock_hand_accuracy_values: list[float]
    teacher_hand_loss_values: list[float]
    teacher_hand_supported_values: list[float]
    teacher_same_family_action_accuracy_values: list[float]
    teacher_same_family_main_play_accuracy_values: list[float]
    teacher_action_margin_mean_values: list[float]
    teacher_action_margin_satisfied_values: list[float]
    teacher_same_family_action_margin_mean_values: list[float]
    teacher_same_family_action_margin_satisfied_values: list[float]
    teacher_public_heuristic_loss_values: list[float]
    teacher_public_heuristic_supported_values: list[float]
    teacher_public_heuristic_top1_mass_values: list[float]
    teacher_public_heuristic_target_entropy_values: list[float]
    teacher_tactical_row_fraction_values: list[float]
    policy_anchor_coef_active_values: list[float]
    policy_anchor_top_action_coef_active_values: list[float]
    policy_anchor_loss_values: list[float]
    policy_anchor_weighted_loss_values: list[float]
    policy_anchor_kl_mean_values: list[float]
    policy_anchor_kl_p95_values: list[float]
    policy_anchor_top_action_loss_values: list[float]
    policy_anchor_top_action_loss_p95_values: list[float]
    policy_anchor_top_action_agreement_values: list[float]

    def section(self) -> dict[str, Any]:
        return {
            "teacher_public_heuristic_coef_active": _window_summary(
                self.teacher_public_heuristic_coef_active_values,
                window=20,
            ),
            "teacher_hand_coef_active": _window_summary(
                self.teacher_hand_coef_active_values,
                window=20,
            ),
            "teacher_aux_loss": _window_summary(self.teacher_aux_loss_values, window=20),
            "teacher_main_play_character_slot_accuracy": _window_summary(
                self.teacher_main_play_slot_accuracy_values,
                window=20,
            ),
            "teacher_hand_accuracy": _window_summary(self.teacher_hand_accuracy_values, window=20),
            "teacher_main_play_character_hand_accuracy": _window_summary(
                self.teacher_main_play_hand_accuracy_values,
                window=20,
            ),
            "teacher_clock_from_hand_accuracy": _window_summary(
                self.teacher_clock_hand_accuracy_values,
                window=20,
            ),
            "teacher_hand_loss": _window_summary(self.teacher_hand_loss_values, window=20),
            "teacher_hand_supported_fraction": _window_summary(self.teacher_hand_supported_values, window=20),
            "teacher_same_family_action_accuracy": _window_summary(
                self.teacher_same_family_action_accuracy_values,
                window=20,
            ),
            "teacher_same_family_main_play_character_accuracy": _window_summary(
                self.teacher_same_family_main_play_accuracy_values,
                window=20,
            ),
            "teacher_action_margin_mean": _window_summary(
                self.teacher_action_margin_mean_values,
                window=20,
            ),
            "teacher_action_margin_satisfied_fraction": _window_summary(
                self.teacher_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_same_family_action_margin_mean": _window_summary(
                self.teacher_same_family_action_margin_mean_values,
                window=20,
            ),
            "teacher_same_family_action_margin_satisfied_fraction": _window_summary(
                self.teacher_same_family_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_public_heuristic_loss": _window_summary(
                self.teacher_public_heuristic_loss_values,
                window=20,
            ),
            "teacher_public_heuristic_supported_fraction": _window_summary(
                self.teacher_public_heuristic_supported_values,
                window=20,
            ),
            "teacher_public_heuristic_top1_mass": _window_summary(
                self.teacher_public_heuristic_top1_mass_values,
                window=20,
            ),
            "teacher_public_heuristic_target_entropy": _window_summary(
                self.teacher_public_heuristic_target_entropy_values,
                window=20,
            ),
            "teacher_tactical_row_fraction_of_total": _window_summary(
                self.teacher_tactical_row_fraction_values,
                window=20,
            ),
            "policy_anchor_coef_active": _window_summary(self.policy_anchor_coef_active_values, window=20),
            "policy_anchor_top_action_coef_active": _window_summary(
                self.policy_anchor_top_action_coef_active_values,
                window=20,
            ),
            "policy_anchor_loss": _window_summary(self.policy_anchor_loss_values, window=20),
            "policy_anchor_weighted_loss": _window_summary(self.policy_anchor_weighted_loss_values, window=20),
            "policy_anchor_kl_mean": _window_summary(self.policy_anchor_kl_mean_values, window=20),
            "policy_anchor_kl_p95": _window_summary(self.policy_anchor_kl_p95_values, window=20),
            "policy_anchor_top_action_loss": _window_summary(self.policy_anchor_top_action_loss_values, window=20),
            "policy_anchor_top_action_loss_p95": _window_summary(
                self.policy_anchor_top_action_loss_p95_values,
                window=20,
            ),
            "policy_anchor_top_action_agreement": _window_summary(
                self.policy_anchor_top_action_agreement_values,
                window=20,
            ),
            "max_teacher_public_heuristic_coef_active": _max_or_none(
                self.teacher_public_heuristic_coef_active_values
            ),
            "max_teacher_hand_coef_active": _max_or_none(self.teacher_hand_coef_active_values),
            "max_teacher_public_heuristic_supported_fraction": _max_or_none(
                self.teacher_public_heuristic_supported_values
            ),
            "max_teacher_hand_supported_fraction": _max_or_none(self.teacher_hand_supported_values),
        }


def build_teacher_guidance_series(
    *,
    records_for_learning: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
) -> TeacherGuidanceSeries:
    return TeacherGuidanceSeries(
        teacher_public_heuristic_coef_active_values=_numeric_values(
            records_for_learning,
            "teacher_public_heuristic_coef_active",
        ),
        teacher_hand_coef_active_values=_numeric_values(records_for_learning, "teacher_hand_coef_active"),
        teacher_aux_loss_values=_numeric_values(records_for_learning, "teacher_aux_loss"),
        teacher_main_play_slot_accuracy_values=_numeric_values(
            records_for_learning,
            "teacher_main_play_character_slot_accuracy",
        ),
        teacher_hand_accuracy_values=_numeric_values(records_for_learning, "teacher_hand_accuracy"),
        teacher_main_play_hand_accuracy_values=_numeric_values(
            records_for_learning,
            "teacher_main_play_character_hand_accuracy",
        ),
        teacher_clock_hand_accuracy_values=_numeric_values(records_for_learning, "teacher_clock_from_hand_accuracy"),
        teacher_hand_loss_values=_numeric_values(records_for_learning, "teacher_hand_loss"),
        teacher_hand_supported_values=_numeric_values(records_for_learning, "teacher_hand_supported_fraction"),
        teacher_same_family_action_accuracy_values=_numeric_values(
            records_for_learning,
            "teacher_same_family_action_accuracy",
        ),
        teacher_same_family_main_play_accuracy_values=_numeric_values(
            records_for_learning,
            "teacher_same_family_main_play_character_accuracy",
        ),
        teacher_action_margin_mean_values=_numeric_values(records_for_learning, "teacher_action_margin_mean"),
        teacher_action_margin_satisfied_values=_numeric_values(
            records_for_learning,
            "teacher_action_margin_satisfied_fraction",
        ),
        teacher_same_family_action_margin_mean_values=_numeric_values(
            records_for_learning,
            "teacher_same_family_action_margin_mean",
        ),
        teacher_same_family_action_margin_satisfied_values=_numeric_values(
            records_for_learning,
            "teacher_same_family_action_margin_satisfied_fraction",
        ),
        teacher_public_heuristic_loss_values=_numeric_values(records_for_learning, "teacher_public_heuristic_loss"),
        teacher_public_heuristic_supported_values=_numeric_values(
            records_for_learning,
            "teacher_public_heuristic_supported_fraction",
        ),
        teacher_public_heuristic_top1_mass_values=_numeric_values(
            records_for_learning,
            "teacher_public_heuristic_top1_mass",
        ),
        teacher_public_heuristic_target_entropy_values=_numeric_values(
            records_for_learning,
            "teacher_public_heuristic_target_entropy",
        ),
        teacher_tactical_row_fraction_values=_fraction_values(
            scalars,
            "collector_teacher_tactical_row_count",
            "collector_total_actions",
        ),
        policy_anchor_coef_active_values=_numeric_values(records_for_learning, "policy_anchor_coef_active"),
        policy_anchor_top_action_coef_active_values=_numeric_values(
            records_for_learning,
            "policy_anchor_top_action_coef_active",
        ),
        policy_anchor_loss_values=_numeric_values(records_for_learning, "policy_anchor_loss"),
        policy_anchor_weighted_loss_values=_numeric_values(records_for_learning, "policy_anchor_weighted_loss"),
        policy_anchor_kl_mean_values=_numeric_values(records_for_learning, "policy_anchor_kl_mean"),
        policy_anchor_kl_p95_values=_numeric_values(records_for_learning, "policy_anchor_kl_p95"),
        policy_anchor_top_action_loss_values=_numeric_values(records_for_learning, "policy_anchor_top_action_loss"),
        policy_anchor_top_action_loss_p95_values=_numeric_values(
            records_for_learning,
            "policy_anchor_top_action_loss_p95",
        ),
        policy_anchor_top_action_agreement_values=_numeric_values(
            records_for_learning,
            "policy_anchor_top_action_agreement",
        ),
    )


def _max_or_none(values: list[float]) -> float | None:
    return None if not values else max(values)


__all__ = ["TeacherGuidanceSeries", "build_teacher_guidance_series"]
