"""In-training paired-swing replay regularizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weiss_rl.replay.trajectory_bc import replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_runner import (
    AuxiliaryReplayBatchContext,
)
from weiss_rl.training.auxiliary_replay_support import (
    trajectory_bc_compatible_training_config,
)
from weiss_rl.training.replay_data import paired_swing_conflict_filter as _conflict_filter
from weiss_rl.training.replay_data.paired_auxiliary_replay import (
    emit_paired_auxiliary_replay_metrics,
    run_due_paired_auxiliary_replay,
)
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

_COMPARE_TO_CHOICES = frozenset({"negative", "top_other"})


@dataclass(slots=True)
class PairedSwingReplayState:
    sampler: TrajectoryBcReplayState
    margin: float
    coef: float
    positive_action_source: str
    negative_action_source: str
    distinct_train_rows: int
    loss_scope: str = "row"
    compare_to: str = "negative"
    conflict_filter_summary: dict[str, Any] | None = None

    @classmethod
    def from_training_config(cls, training_config: Any, *, repo_root: Path) -> PairedSwingReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "paired_swing_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "paired_swing_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        positive_action_source = _conflict_filter.normalize_paired_swing_action_source(
            getattr(structured_aux, "paired_swing_positive_action_source", "teacher_action"),
            field_name="paired_swing_positive_action_source",
        )
        negative_action_source = _conflict_filter.normalize_paired_swing_action_source(
            getattr(structured_aux, "paired_swing_negative_action_source", "actions"),
            field_name="paired_swing_negative_action_source",
        )
        if positive_action_source == negative_action_source:
            raise ValueError("paired_swing_positive_action_source and paired_swing_negative_action_source must differ")
        loss_scope = str(getattr(structured_aux, "paired_swing_loss_scope", "row")).strip().lower()
        if loss_scope not in {"row", "episode_mean", "label_mean"}:
            raise ValueError("paired_swing_loss_scope must be one of: episode_mean, label_mean, row")
        compare_to = str(getattr(structured_aux, "paired_swing_compare_to", "negative")).strip().lower()
        if compare_to not in _COMPARE_TO_CHOICES:
            raise ValueError("paired_swing_compare_to must be one of: negative, top_other")
        conflict_filter = str(getattr(structured_aux, "paired_swing_conflict_filter", "none")).strip().lower()
        if conflict_filter not in _conflict_filter.PAIRED_SWING_CONFLICT_FILTERS:
            raise ValueError("paired_swing_conflict_filter must be one of: current_state, history, none")
        margin = float(getattr(structured_aux, "paired_swing_margin", 0.35))
        if margin < 0.0:
            raise ValueError("paired_swing_margin must be >= 0.0")
        coef = float(getattr(structured_aux, "paired_swing_coef", 0.08))
        if coef < 0.0:
            raise ValueError("paired_swing_coef must be >= 0.0")

        sampler = TrajectoryBcReplayState.from_training_config(
            trajectory_bc_compatible_training_config(
                structured_aux=structured_aux,
                dataset_path_text=dataset_path_text,
                every_updates=every_updates,
                field_prefix="paired_swing",
                seed_default=20260519,
                include_focus_fields=True,
            ),
            repo_root=repo_root,
        )
        if sampler is None:
            return None
        conflict_filter_summary: dict[str, Any] | None = None
        if conflict_filter != "none":
            filtered_dataset, conflict_filter_summary = _conflict_filter.filter_paired_swing_conflict_rows(
                sampler.dataset,
                mode=conflict_filter,
                positive_action_source=positive_action_source,
                negative_action_source=negative_action_source,
            )
            sampler.dataset = filtered_dataset
            sampler.order = sampler.rng.permutation(filtered_dataset.episode_count)
            sampler.cursor = 0
            sampler.focus_cursor = 0
            sampler.nonfocus_cursor = 0
        distinct_train_rows = _conflict_filter.paired_swing_distinct_train_row_count(
            sampler.dataset,
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
        )
        if distinct_train_rows <= 0:
            raise ValueError(
                "paired-swing dataset has no trainable rows where positive and negative actions differ: "
                f"{dataset_path_text}"
            )
        return cls(
            sampler=sampler,
            margin=margin,
            coef=coef,
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
            distinct_train_rows=distinct_train_rows,
            loss_scope=loss_scope,
            compare_to=compare_to,
            conflict_filter_summary=conflict_filter_summary,
        )


def maybe_run_paired_swing_replay(
    *,
    state: PairedSwingReplayState | None,
    learner: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured paired-swing auxiliary steps after an RL update."""

    def make_update_batch(updater: Any) -> Any:
        def update_batch(batch: dict[str, Any], _context: AuxiliaryReplayBatchContext) -> dict[str, float]:
            assert state is not None
            return dict(
                updater(
                    batch,
                    margin=float(state.margin),
                    coef=float(state.coef),
                    positive_action_source=state.positive_action_source,
                    negative_action_source=state.negative_action_source,
                    loss_scope=state.loss_scope,
                    compare_to=state.compare_to,
                )
            )

        return update_batch

    replay_result = run_due_paired_auxiliary_replay(
        state=state,
        learner=learner,
        device=device,
        update_count=update_count,
        updater_method_name="paired_swing_update",
        updater_error_message="learner does not support paired_swing_update",
        make_update_batch=make_update_batch,
        batch_factory=replay_trajectory_bc_batch,
        use_opponent_context=True,
    )
    if replay_result is None:
        return

    assert state is not None
    emit_paired_auxiliary_replay_metrics(
        latest_metrics,
        prefix="paired_swing_replay",
        replay_result=replay_result,
        static_metrics=_paired_swing_replay_static_metrics(state),
        include_focus=True,
        include_context=True,
    )


def _paired_swing_replay_static_metrics(state: PairedSwingReplayState) -> dict[str, float]:
    metrics = {
        "paired_swing_replay_dataset_distinct_train_rows": float(state.distinct_train_rows),
        "paired_swing_replay_margin": float(state.margin),
        "paired_swing_replay_coef": float(state.coef),
        "paired_swing_replay_loss_scope_episode_mean": 1.0 if state.loss_scope == "episode_mean" else 0.0,
        "paired_swing_replay_loss_scope_label_mean": 1.0 if state.loss_scope == "label_mean" else 0.0,
    }
    if state.compare_to == "top_other":
        metrics["paired_swing_replay_compare_to_top_other"] = 1.0
    if state.conflict_filter_summary is not None:
        summary = state.conflict_filter_summary
        metrics.update(
            {
                "paired_swing_replay_conflict_filter_active": 1.0,
                "paired_swing_replay_conflict_filter_dropped_rows": float(summary.get("dropped_train_rows", 0)),
                "paired_swing_replay_conflict_filter_kept_rows": float(summary.get("kept_train_rows", 0)),
                "paired_swing_replay_conflict_filter_conflict_keys": float(summary.get("conflict_key_count", 0)),
            }
        )
    if state.positive_action_source == "teacher_action":
        metrics["paired_swing_replay_positive_source_teacher"] = 1.0
    if state.negative_action_source == "teacher_action":
        metrics["paired_swing_replay_negative_source_teacher"] = 1.0
    return metrics


__all__ = [
    "PairedSwingReplayState",
    "maybe_run_paired_swing_replay",
]
