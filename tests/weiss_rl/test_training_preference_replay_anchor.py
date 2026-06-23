from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.training.replay_data.training_replay_states import (
    TrainingReplayStates,
    reset_policy_anchor_for_fresh_preference_replay,
)


def test_fresh_preference_replay_resets_policy_anchor_once() -> None:
    calls: list[dict[str, object]] = []
    learner = SimpleNamespace(reset_policy_anchor_to_current_model=lambda **kwargs: calls.append(dict(kwargs)))
    replay_states = TrainingReplayStates(
        trajectory_bc=None,
        paired_swing=None,
        paired_outcome_preference=object(),
    )

    reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=replay_states,
        resume_state=None,
    )

    assert calls == [{"force": True}]


def test_preference_replay_anchor_reset_skips_resume_and_disabled_replay() -> None:
    learner = SimpleNamespace(
        reset_policy_anchor_to_current_model=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("policy anchor should only reset for fresh preference replay")
        )
    )

    reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=TrainingReplayStates(
            trajectory_bc=None,
            paired_swing=None,
            paired_outcome_preference=None,
        ),
        resume_state=None,
    )
    reset_policy_anchor_for_fresh_preference_replay(
        learner=learner,
        replay_states=TrainingReplayStates(
            trajectory_bc=None,
            paired_swing=None,
            paired_outcome_preference=object(),
        ),
        resume_state={},
    )


def test_fresh_preference_replay_requires_policy_anchor_support() -> None:
    with pytest.raises(ValueError, match="policy-anchor support"):
        reset_policy_anchor_for_fresh_preference_replay(
            learner=object(),
            replay_states=TrainingReplayStates(
                trajectory_bc=None,
                paired_swing=None,
                paired_outcome_preference=object(),
            ),
            resume_state=None,
        )
