"""Shared helpers for structured teacher-auxiliary loss implementations."""

from __future__ import annotations

from torch import Tensor


def empty_structured_teacher_metrics() -> dict[str, float]:
    return {
        "teacher_active_fraction": 0.0,
        "teacher_valid_fraction": 0.0,
        "teacher_main_play_character_fraction": 0.0,
        "teacher_main_move_fraction": 0.0,
        "teacher_attack_fraction": 0.0,
        "teacher_family_accuracy": 0.0,
        "teacher_slot_accuracy": 0.0,
        "teacher_main_play_character_slot_accuracy": 0.0,
        "teacher_hand_accuracy": 0.0,
        "teacher_main_play_character_hand_accuracy": 0.0,
        "teacher_clock_from_hand_accuracy": 0.0,
        "teacher_move_source_accuracy": 0.0,
        "teacher_attack_type_accuracy": 0.0,
        "teacher_action_accuracy": 0.0,
        "teacher_same_family_action_accuracy": 0.0,
        "teacher_same_family_main_play_character_accuracy": 0.0,
        "teacher_same_family_main_move_accuracy": 0.0,
        "teacher_family_loss": 0.0,
        "teacher_slot_loss": 0.0,
        "teacher_hand_loss": 0.0,
        "teacher_hand_supported_fraction": 0.0,
        "teacher_move_source_loss": 0.0,
        "teacher_move_source_supported_fraction": 0.0,
        "teacher_attack_type_loss": 0.0,
        "teacher_action_loss": 0.0,
        "teacher_action_supported_fraction": 0.0,
        "teacher_action_margin_loss": 0.0,
        "teacher_action_margin_supported_fraction": 0.0,
        "teacher_action_margin_mean": 0.0,
        "teacher_action_margin_satisfied_fraction": 0.0,
        "teacher_same_family_action_margin_loss": 0.0,
        "teacher_same_family_action_margin_supported_fraction": 0.0,
        "teacher_same_family_action_margin_mean": 0.0,
        "teacher_same_family_action_margin_satisfied_fraction": 0.0,
        "teacher_same_family_action_loss": 0.0,
        "teacher_same_family_action_supported_fraction": 0.0,
        "teacher_public_heuristic_loss": 0.0,
        "teacher_public_heuristic_supported_fraction": 0.0,
        "teacher_public_heuristic_top1_mass": 0.0,
        "teacher_public_heuristic_target_entropy": 0.0,
        "teacher_public_nonpass_over_pass_loss": 0.0,
        "teacher_public_nonpass_over_pass_supported_fraction": 0.0,
        "teacher_public_nonpass_over_pass_margin_mean": 0.0,
        "teacher_public_nonpass_over_pass_satisfied_fraction": 0.0,
        "teacher_aux_loss": 0.0,
    }


def record_teacher_family_coverage(
    metrics: dict[str, float],
    *,
    active_rows: Tensor,
    flat_teacher_family: Tensor,
    flat_teacher_valid: Tensor,
    play_family_id: int,
    move_family_id: int,
    attack_family_id: int,
) -> None:
    active_total = float(active_rows.float().sum().item())
    metrics["teacher_active_fraction"] = active_total / max(float(active_rows.numel()), 1.0)
    if active_total <= 0.0:
        return
    family_rows = active_rows & flat_teacher_valid & (flat_teacher_family >= 0)
    if play_family_id >= 0:
        metrics["teacher_main_play_character_fraction"] = float(
            ((family_rows & (flat_teacher_family == play_family_id)).float().sum().item()) / active_total
        )
    if move_family_id >= 0:
        metrics["teacher_main_move_fraction"] = float(
            ((family_rows & (flat_teacher_family == move_family_id)).float().sum().item()) / active_total
        )
    if attack_family_id >= 0:
        metrics["teacher_attack_fraction"] = float(
            ((family_rows & (flat_teacher_family == attack_family_id)).float().sum().item()) / active_total
        )


__all__ = [
    "empty_structured_teacher_metrics",
    "record_teacher_family_coverage",
]
