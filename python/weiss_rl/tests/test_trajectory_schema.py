from __future__ import annotations

import pytest

from weiss_rl.trajectory.schema import (
    TrajectoryDebug,
    TrajectoryStep,
    legal_storage_fields,
)


def test_trajectory_step_keeps_behavior_logp_with_step_fields() -> None:
    step = TrajectoryStep(
        obs=[1, 2, 3],
        to_play_seat=1,
        decision_id=7,
        action=51,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        episode_seed=123,
        episode_key=b"episode-key",
        behavior_logp=-0.75,
    )

    assert step.behavior_logp == -0.75
    assert step.episode_key == b"episode-key"


def test_legal_storage_fields_match_the_declared_legal_repr() -> None:
    assert legal_storage_fields("ids_offsets") == ("legal_ids", "legal_offsets")
    assert legal_storage_fields("mask") == ("legal_mask",)
    assert legal_storage_fields("none") == ()


def test_k_raw_decisions_is_required_for_folded_steps() -> None:
    debug = TrajectoryDebug()

    with pytest.raises(ValueError, match="k_raw_decisions is required"):
        debug.validate(step_definition="learner_turn_env")


def test_k_raw_decisions_must_be_positive_when_recorded() -> None:
    debug = TrajectoryDebug(k_raw_decisions=0)

    with pytest.raises(ValueError, match=">= 1"):
        debug.validate(step_definition="decision_boundary")


def test_k_raw_decisions_is_optional_for_decision_boundary_steps() -> None:
    TrajectoryDebug().validate(step_definition="decision_boundary")
    TrajectoryDebug(k_raw_decisions=2).validate(step_definition="learner_turn_env")
