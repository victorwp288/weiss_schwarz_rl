"""Replay-state construction and anchor lifecycle for training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.training.replay_data.paired_outcome_preference_replay import PairedOutcomePreferenceReplayState
from weiss_rl.training.replay_data.paired_swing_replay import PairedSwingReplayState
from weiss_rl.training.replay_data.trajectory_bc_sampling import TrajectoryBcReplayState


@dataclass(frozen=True, slots=True)
class TrainingReplayStates:
    trajectory_bc: TrajectoryBcReplayState | None
    paired_swing: PairedSwingReplayState | None
    paired_outcome_preference: PairedOutcomePreferenceReplayState | None

    def has_fresh_preference_anchor_requirement(self) -> bool:
        return self.paired_outcome_preference is not None


def training_replay_states_from_config(training_config: Any, *, repo_root: Path) -> TrainingReplayStates:
    return TrainingReplayStates(
        trajectory_bc=TrajectoryBcReplayState.from_training_config(
            training_config,
            repo_root=repo_root,
        ),
        paired_swing=PairedSwingReplayState.from_training_config(
            training_config,
            repo_root=repo_root,
        ),
        paired_outcome_preference=PairedOutcomePreferenceReplayState.from_training_config(
            training_config,
            repo_root=repo_root,
        ),
    )


def reset_policy_anchor_to_current_model(learner: Any) -> None:
    reset = getattr(learner, "reset_policy_anchor_to_current_model", None)
    if not callable(reset):
        raise ValueError("paired outcome preference replay requires learner policy-anchor support")
    try:
        reset(force=True)
    except TypeError:
        reset()


def reset_policy_anchor_for_fresh_preference_replay(
    *,
    learner: Any,
    replay_states: TrainingReplayStates,
    resume_state: Mapping[str, Any] | None,
) -> None:
    if not replay_states.has_fresh_preference_anchor_requirement() or resume_state is not None:
        return
    reset_policy_anchor_to_current_model(learner)


__all__ = [
    "TrainingReplayStates",
    "reset_policy_anchor_for_fresh_preference_replay",
    "reset_policy_anchor_to_current_model",
    "training_replay_states_from_config",
]
