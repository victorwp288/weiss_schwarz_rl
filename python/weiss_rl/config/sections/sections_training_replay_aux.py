"""Replay-data auxiliary parser for structured training config."""

from __future__ import annotations

from typing import Any

from weiss_rl.config.loading.parsing_utils import require_bool, require_choice, require_float, require_int
from weiss_rl.config.sections.sections_training_schema import (
    TRAINING_PAIRED_OUTCOME_PREFERENCE_AGGREGATIONS,
    TRAINING_PAIRED_SWING_COMPARE_TO,
    TRAINING_PAIRED_SWING_CONFLICT_FILTERS,
    TRAINING_PAIRED_SWING_LOSS_SCOPES,
)

from .sections_training_aux_helpers import require_nonnegative_float, require_positive_float
from .sections_training_focus import (
    paired_swing_action_source,
    paired_swing_focus_fraction,
    paired_swing_focus_groups,
    paired_swing_focus_source_labels,
    trajectory_bc_focus_fraction,
    trajectory_bc_focus_groups,
    trajectory_bc_focus_source_labels,
    validate_paired_swing_focus_contract,
    validate_trajectory_bc_focus_contract,
)


def parse_structured_aux_replay_data_settings(structured_aux: dict[str, Any]) -> dict[str, Any]:
    """Parse replay-dataset auxiliary settings for `TrainingStructuredAuxConfig`."""

    trajectory_bc_focus_labels = trajectory_bc_focus_source_labels(structured_aux)
    trajectory_bc_focus_value = trajectory_bc_focus_fraction(structured_aux)
    trajectory_bc_focus_group_values = trajectory_bc_focus_groups(structured_aux)
    validate_trajectory_bc_focus_contract(
        source_labels=trajectory_bc_focus_labels,
        fraction=trajectory_bc_focus_value,
        groups=trajectory_bc_focus_group_values,
    )

    paired_swing_focus_labels = paired_swing_focus_source_labels(structured_aux)
    paired_swing_focus_value = paired_swing_focus_fraction(structured_aux)
    paired_swing_focus_group_values = paired_swing_focus_groups(structured_aux)
    validate_paired_swing_focus_contract(
        source_labels=paired_swing_focus_labels,
        fraction=paired_swing_focus_value,
        groups=paired_swing_focus_group_values,
    )

    paired_swing_positive_source = paired_swing_action_source(
        structured_aux,
        key="paired_swing_positive_action_source",
        default="teacher_action",
    )
    paired_swing_negative_source = paired_swing_action_source(
        structured_aux,
        key="paired_swing_negative_action_source",
        default="actions",
    )
    if paired_swing_positive_source == paired_swing_negative_source:
        raise ValueError(
            "training.structured_aux.paired_swing_positive_action_source and "
            "paired_swing_negative_action_source must differ"
        )

    return {
        "trajectory_bc_dataset_path": str(structured_aux.get("trajectory_bc_dataset_path", "")).strip(),
        "trajectory_bc_every_updates": require_int(
            structured_aux.get("trajectory_bc_every_updates", 0),
            field_name="training.structured_aux.trajectory_bc_every_updates",
            minimum=0,
        ),
        "trajectory_bc_aux_updates": require_int(
            structured_aux.get("trajectory_bc_aux_updates", 1),
            field_name="training.structured_aux.trajectory_bc_aux_updates",
            minimum=1,
        ),
        "trajectory_bc_batch_episodes": require_int(
            structured_aux.get("trajectory_bc_batch_episodes", 8),
            field_name="training.structured_aux.trajectory_bc_batch_episodes",
            minimum=1,
        ),
        "trajectory_bc_seed": require_int(
            structured_aux.get("trajectory_bc_seed", 20260516),
            field_name="training.structured_aux.trajectory_bc_seed",
            minimum=0,
        ),
        "trajectory_bc_focus_source_labels": trajectory_bc_focus_labels,
        "trajectory_bc_focus_fraction": trajectory_bc_focus_value,
        "trajectory_bc_focus_groups": trajectory_bc_focus_group_values,
        "trajectory_bc_teacher_family_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_family_coef", 0.05),
            field_name="training.structured_aux.trajectory_bc_teacher_family_coef",
        ),
        "trajectory_bc_teacher_slot_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_slot_coef", 0.05),
            field_name="training.structured_aux.trajectory_bc_teacher_slot_coef",
        ),
        "trajectory_bc_teacher_move_source_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_move_source_coef", 0.02),
            field_name="training.structured_aux.trajectory_bc_teacher_move_source_coef",
        ),
        "trajectory_bc_teacher_attack_type_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_attack_type_coef", 0.02),
            field_name="training.structured_aux.trajectory_bc_teacher_attack_type_coef",
        ),
        "trajectory_bc_teacher_action_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_action_coef", 0.20),
            field_name="training.structured_aux.trajectory_bc_teacher_action_coef",
        ),
        "trajectory_bc_teacher_same_family_action_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_same_family_action_coef", 0.60),
            field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_coef",
        ),
        "trajectory_bc_teacher_same_family_action_margin_coef": require_float(
            structured_aux.get("trajectory_bc_teacher_same_family_action_margin_coef", 0.10),
            field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_margin_coef",
        ),
        "trajectory_bc_teacher_same_family_action_margin": require_float(
            structured_aux.get("trajectory_bc_teacher_same_family_action_margin", 0.5),
            field_name="training.structured_aux.trajectory_bc_teacher_same_family_action_margin",
        ),
        "paired_swing_dataset_path": str(structured_aux.get("paired_swing_dataset_path", "")).strip(),
        "paired_swing_every_updates": require_int(
            structured_aux.get("paired_swing_every_updates", 0),
            field_name="training.structured_aux.paired_swing_every_updates",
            minimum=0,
        ),
        "paired_swing_aux_updates": require_int(
            structured_aux.get("paired_swing_aux_updates", 1),
            field_name="training.structured_aux.paired_swing_aux_updates",
            minimum=1,
        ),
        "paired_swing_batch_episodes": require_int(
            structured_aux.get("paired_swing_batch_episodes", 8),
            field_name="training.structured_aux.paired_swing_batch_episodes",
            minimum=1,
        ),
        "paired_swing_seed": require_int(
            structured_aux.get("paired_swing_seed", 20260519),
            field_name="training.structured_aux.paired_swing_seed",
            minimum=0,
        ),
        "paired_swing_focus_source_labels": paired_swing_focus_labels,
        "paired_swing_focus_fraction": paired_swing_focus_value,
        "paired_swing_focus_groups": paired_swing_focus_group_values,
        "paired_swing_margin": require_nonnegative_float(
            structured_aux,
            "paired_swing_margin",
            0.35,
            field_name="training.structured_aux.paired_swing_margin",
        ),
        "paired_swing_coef": require_nonnegative_float(
            structured_aux,
            "paired_swing_coef",
            0.08,
            field_name="training.structured_aux.paired_swing_coef",
        ),
        "paired_swing_positive_action_source": paired_swing_positive_source,
        "paired_swing_negative_action_source": paired_swing_negative_source,
        "paired_swing_conflict_filter": require_choice(
            structured_aux.get("paired_swing_conflict_filter", "none"),
            field_name="training.structured_aux.paired_swing_conflict_filter",
            allowed=TRAINING_PAIRED_SWING_CONFLICT_FILTERS,
        ),
        "paired_swing_loss_scope": require_choice(
            structured_aux.get("paired_swing_loss_scope", "row"),
            field_name="training.structured_aux.paired_swing_loss_scope",
            allowed=TRAINING_PAIRED_SWING_LOSS_SCOPES,
        ),
        "paired_swing_compare_to": require_choice(
            structured_aux.get("paired_swing_compare_to", "negative"),
            field_name="training.structured_aux.paired_swing_compare_to",
            allowed=TRAINING_PAIRED_SWING_COMPARE_TO,
        ),
        "paired_outcome_preference_dataset_path": str(
            structured_aux.get("paired_outcome_preference_dataset_path", "")
        ).strip(),
        "paired_outcome_preference_every_updates": require_int(
            structured_aux.get("paired_outcome_preference_every_updates", 0),
            field_name="training.structured_aux.paired_outcome_preference_every_updates",
            minimum=0,
        ),
        "paired_outcome_preference_aux_updates": require_int(
            structured_aux.get("paired_outcome_preference_aux_updates", 1),
            field_name="training.structured_aux.paired_outcome_preference_aux_updates",
            minimum=1,
        ),
        "paired_outcome_preference_batch_episodes": require_int(
            structured_aux.get("paired_outcome_preference_batch_episodes", 8),
            field_name="training.structured_aux.paired_outcome_preference_batch_episodes",
            minimum=1,
        ),
        "paired_outcome_preference_seed": require_int(
            structured_aux.get("paired_outcome_preference_seed", 20260520),
            field_name="training.structured_aux.paired_outcome_preference_seed",
            minimum=0,
        ),
        "paired_outcome_preference_coef": require_nonnegative_float(
            structured_aux,
            "paired_outcome_preference_coef",
            0.05,
            field_name="training.structured_aux.paired_outcome_preference_coef",
        ),
        "paired_outcome_preference_beta": require_positive_float(
            structured_aux,
            "paired_outcome_preference_beta",
            0.1,
            field_name="training.structured_aux.paired_outcome_preference_beta",
        ),
        "paired_outcome_preference_aggregation": require_choice(
            structured_aux.get("paired_outcome_preference_aggregation", "mean"),
            field_name="training.structured_aux.paired_outcome_preference_aggregation",
            allowed=TRAINING_PAIRED_OUTCOME_PREFERENCE_AGGREGATIONS,
        ),
        "paired_outcome_preference_group_balance": require_bool(
            structured_aux.get("paired_outcome_preference_group_balance", False),
            field_name="training.structured_aux.paired_outcome_preference_group_balance",
        ),
    }


__all__ = ["parse_structured_aux_replay_data_settings"]
