"""Schedule application for one canonical learner update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.config import StackConfig


@dataclass(frozen=True, slots=True)
class TrainingUpdateScheduleResult:
    update_count: int
    metrics: dict[str, float]


def schedule_update_count_for_next_update(
    *,
    learner_update_count: int,
    init_schedule_offset_updates: int,
) -> int:
    """Map fresh-run local updates onto source-checkpoint schedule time."""

    return max(0, int(init_schedule_offset_updates)) + max(0, int(learner_update_count)) + 1


def apply_training_update_schedule(
    *,
    learner: Any,
    model: Any,
    stack: StackConfig,
    training_config: Any,
    init_schedule_offset_updates: int,
    apply_guidance_schedule_for_next_update: Any,
    entropy_coef_for_next_update: Any,
) -> TrainingUpdateScheduleResult:
    schedule_update_count = schedule_update_count_for_next_update(
        learner_update_count=int(learner.update_count),
        init_schedule_offset_updates=init_schedule_offset_updates,
    )
    metrics = apply_guidance_schedule_for_next_update(
        learner=learner,
        model=model,
        stack=stack,
        update_count=schedule_update_count,
    )
    metrics["guidance_schedule_update_count"] = float(schedule_update_count)
    if init_schedule_offset_updates > 0:
        metrics["init_schedule_offset_updates"] = float(init_schedule_offset_updates)
    learner.set_entropy_coef(entropy_coef_for_next_update(training_config, update_count=schedule_update_count))
    return TrainingUpdateScheduleResult(update_count=schedule_update_count, metrics=metrics)


__all__ = [
    "TrainingUpdateScheduleResult",
    "apply_training_update_schedule",
    "schedule_update_count_for_next_update",
]
