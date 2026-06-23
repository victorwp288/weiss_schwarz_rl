from __future__ import annotations

import pytest
from weiss_rl.runtime.components.teacher_heuristic_mixin import QueueRuntimeTeacherHeuristicMixin
from weiss_rl.runtime.components.teacher_labels import (
    selected_teacher_label_profile,
    teacher_guidance_active_for_collection,
)


def test_teacher_guidance_active_for_collection_preserves_aux_mode_rules() -> None:
    assert not teacher_guidance_active_for_collection(
        enabled=False,
        teacher_aux_mode="always",
        warmstart_updates=3,
        current_learner_update=0,
    )
    assert not teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="off",
        warmstart_updates=3,
        current_learner_update=0,
    )
    assert teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="always",
        warmstart_updates=0,
        current_learner_update=99,
    )
    assert teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="warmstart_only",
        warmstart_updates=2,
        current_learner_update=1,
    )
    assert not teacher_guidance_active_for_collection(
        enabled=True,
        teacher_aux_mode="warmstart_only",
        warmstart_updates=2,
        current_learner_update=2,
    )


def test_selected_teacher_label_profile_tracks_cycle_mode_until_end_update() -> None:
    profiles = ("base", "aggressive", "control")

    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=0, end_updates=3) == "base"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=1, end_updates=3) == "aggressive"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=2, end_updates=3) == "control"
    assert selected_teacher_label_profile(profiles, profile_mode="cycle", update_count=4, end_updates=3) == "base"
    assert selected_teacher_label_profile(profiles, profile_mode="mixture", update_count=1, end_updates=-1) == "base"

    with pytest.raises(ValueError, match="unsupported profiles"):
        selected_teacher_label_profile(("unknown",), profile_mode="cycle", update_count=0, end_updates=-1)


def test_teacher_heuristic_mixin_selects_profiled_teacher_policy_for_labels() -> None:
    class Runtime(QueueRuntimeTeacherHeuristicMixin):
        _teacher_policy = "base-policy"
        _teacher_policy_by_profile = {
            "base": "base-policy",
            "aggressive": "aggressive-policy",
            "control": "control-policy",
        }
        _teacher_label_profiles = ("base", "aggressive", "control")
        _teacher_label_profile_mode = "cycle"
        _teacher_label_profiles_end_updates = 150
        _current_learner_update = 2

    assert Runtime()._teacher_label_policy_for_current_update() == "control-policy"
