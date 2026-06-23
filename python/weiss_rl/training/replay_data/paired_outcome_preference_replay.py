"""In-training paired outcome preference replay regularizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_runner import (
    AuxiliaryReplayBatchContext,
)
from weiss_rl.training.auxiliary_replay_support import (
    trajectory_bc_compatible_training_config,
)
from weiss_rl.training.replay_data import paired_outcome_preference_dataset as _preference_dataset
from weiss_rl.training.replay_data.paired_auxiliary_replay import (
    emit_paired_auxiliary_replay_metrics,
    run_due_paired_auxiliary_replay,
)
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState

_AGGREGATIONS = frozenset({"mean", "sum"})


@dataclass(slots=True)
class PairedOutcomePreferenceReplayState:
    sampler: TrajectoryBcReplayState
    beta: float
    coef: float
    aggregation: str
    group_balance: bool
    complete_pair_count: int

    @classmethod
    def from_training_config(
        cls, training_config: Any, *, repo_root: Path
    ) -> PairedOutcomePreferenceReplayState | None:
        structured_aux = training_config.structured_aux
        dataset_path_text = str(getattr(structured_aux, "paired_outcome_preference_dataset_path", "")).strip()
        every_updates = int(getattr(structured_aux, "paired_outcome_preference_every_updates", 0))
        if not dataset_path_text or every_updates <= 0:
            return None
        beta = float(getattr(structured_aux, "paired_outcome_preference_beta", 0.1))
        if beta <= 0.0:
            raise ValueError("paired_outcome_preference_beta must be > 0.0")
        coef = float(getattr(structured_aux, "paired_outcome_preference_coef", 0.05))
        if coef < 0.0:
            raise ValueError("paired_outcome_preference_coef must be >= 0.0")
        aggregation = str(getattr(structured_aux, "paired_outcome_preference_aggregation", "mean")).strip().lower()
        if aggregation not in _AGGREGATIONS:
            raise ValueError("paired_outcome_preference_aggregation must be one of: mean, sum")
        group_balance = bool(getattr(structured_aux, "paired_outcome_preference_group_balance", False))

        sampler = TrajectoryBcReplayState.from_training_config(
            trajectory_bc_compatible_training_config(
                structured_aux=structured_aux,
                dataset_path_text=dataset_path_text,
                every_updates=every_updates,
                field_prefix="paired_outcome_preference",
                seed_default=20260520,
                include_focus_fields=False,
            ),
            repo_root=repo_root,
        )
        if sampler is None:
            return None
        complete_pair_count = _preference_dataset.paired_outcome_preference_complete_pair_count(sampler.dataset)
        if complete_pair_count <= 0:
            raise ValueError(
                f"paired outcome preference dataset has no complete preferred/rejected pairs: {dataset_path_text}"
            )
        return cls(
            sampler=sampler,
            beta=beta,
            coef=coef,
            aggregation=aggregation,
            group_balance=group_balance,
            complete_pair_count=complete_pair_count,
        )


def maybe_run_paired_outcome_preference_replay(
    *,
    state: PairedOutcomePreferenceReplayState | None,
    learner: Any,
    device: torch.device,
    update_count: int,
    latest_metrics: dict[str, float],
) -> None:
    """Run configured paired outcome preference auxiliary steps after an RL update."""

    def make_update_batch(updater: Any) -> Any:
        def update_batch(batch: dict[str, Any], context: AuxiliaryReplayBatchContext) -> dict[str, float]:
            assert state is not None
            preference_group_indices = _preference_dataset.preference_group_indices_for_episodes(
                state.sampler.dataset,
                episode_indices=context.episode_indices,
            )
            if preference_group_indices is not None:
                batch["preference_group_id"] = np.broadcast_to(
                    np.asarray(preference_group_indices, dtype=np.int64).reshape(1, -1),
                    np.asarray(batch["actions"]).shape,
                ).copy()
            return dict(
                updater(
                    batch,
                    beta=float(state.beta),
                    coef=float(state.coef),
                    aggregation=state.aggregation,
                    group_balance=bool(state.group_balance),
                )
            )

        return update_batch

    replay_result = run_due_paired_auxiliary_replay(
        state=state,
        learner=learner,
        device=device,
        update_count=update_count,
        updater_method_name="paired_outcome_preference_update",
        updater_error_message="learner does not support paired_outcome_preference_update",
        make_update_batch=make_update_batch,
        batch_factory=replay_trajectory_bc_batch,
        use_opponent_context=True,
    )
    if replay_result is None:
        return

    assert state is not None
    emit_paired_auxiliary_replay_metrics(
        latest_metrics,
        prefix="paired_outcome_preference_replay",
        replay_result=replay_result,
        static_metrics=_paired_outcome_preference_replay_static_metrics(state),
        include_context=True,
    )


def _paired_outcome_preference_replay_static_metrics(state: PairedOutcomePreferenceReplayState) -> dict[str, float]:
    return {
        "paired_outcome_preference_replay_complete_pair_count": float(state.complete_pair_count),
        "paired_outcome_preference_replay_beta": float(state.beta),
        "paired_outcome_preference_replay_coef": float(state.coef),
        "paired_outcome_preference_replay_aggregation_sum": 1.0 if state.aggregation == "sum" else 0.0,
        "paired_outcome_preference_replay_group_balance": 1.0 if state.group_balance else 0.0,
    }


__all__ = [
    "PairedOutcomePreferenceReplayState",
    "maybe_run_paired_outcome_preference_replay",
]
