from __future__ import annotations

from types import SimpleNamespace

import pytest

from weiss_rl.training.guidance import (
    apply_guidance_schedule_for_next_update,
    entropy_coef_for_next_update,
    model_guidance_payload,
    restore_model_guidance_from_payload,
    teacher_public_heuristic_coef_for_next_update,
    teacher_supervised_coef_scale_for_next_update,
)


class _FakeLearner:
    def __init__(self) -> None:
        self.public_heuristic_coef: float | None = None
        self.teacher_aux_coefs: dict[str, float] = {}

    def set_teacher_aux_coefs(self, **kwargs: float) -> None:
        self.teacher_aux_coefs.update({key: float(value) for key, value in kwargs.items()})
        self.public_heuristic_coef = self.teacher_aux_coefs.get("public_heuristic")


class _FakeGuidedModel:
    def __init__(self, *, learner_scale: float = 0.2, actor_scale: float = 0.7) -> None:
        self.learner_scale = learner_scale
        self.actor_scale = actor_scale

    def get_public_heuristic_logit_bias_scale(self, *, scoring_mode: str) -> float:
        if scoring_mode == "actor":
            return self.actor_scale
        if scoring_mode == "learner":
            return self.learner_scale
        raise ValueError(scoring_mode)

    def set_public_heuristic_logit_bias_scale(self, value: float, *, actor_value: float | None = None) -> None:
        self.learner_scale = float(value)
        if actor_value is not None:
            self.actor_scale = float(actor_value)


def test_guidance_schedules_preserve_linear_training_values() -> None:
    training = SimpleNamespace(
        entropy_coef=0.2,
        entropy_anneal_to=0.02,
        entropy_anneal_steps_updates=10,
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_final_coef=0.0,
        teacher_public_heuristic_start_updates=10,
        teacher_public_heuristic_end_updates=20,
        teacher_supervised_final_scale=0.25,
        teacher_supervised_start_updates=40,
        teacher_supervised_end_updates=60,
    )

    assert entropy_coef_for_next_update(training, update_count=5) == pytest.approx(0.11)
    assert teacher_public_heuristic_coef_for_next_update(training, update_count=15) == pytest.approx(0.5)
    assert teacher_supervised_coef_scale_for_next_update(training, update_count=50) == pytest.approx(0.625)


def test_guidance_schedule_updates_learner_and_model_without_actor_scale_drift() -> None:
    training = SimpleNamespace(
        teacher_public_heuristic_coef=0.8,
        teacher_public_heuristic_final_coef=0.2,
        teacher_public_heuristic_start_updates=0,
        teacher_public_heuristic_end_updates=10,
        teacher_family_coef=0.03,
        teacher_slot_coef=0.08,
        teacher_hand_coef=0.07,
        teacher_move_source_coef=0.02,
        teacher_attack_type_coef=0.04,
        teacher_action_coef=0.05,
        teacher_same_family_action_coef=0.12,
        teacher_action_margin_coef=0.10,
        teacher_same_family_action_margin_coef=0.06,
        teacher_supervised_final_scale=0.25,
        teacher_supervised_start_updates=0,
        teacher_supervised_end_updates=10,
    )
    model_config = SimpleNamespace(
        public_heuristic_logit_bias_scale=1.0,
        public_heuristic_logit_bias_final_scale=0.0,
        public_heuristic_logit_bias_start_updates=0,
        public_heuristic_logit_bias_end_updates=10,
    )
    stack = SimpleNamespace(config=SimpleNamespace(training=training, model=model_config))
    learner = _FakeLearner()
    model = _FakeGuidedModel(learner_scale=0.9, actor_scale=0.4)

    metrics = apply_guidance_schedule_for_next_update(
        learner=learner,
        model=model,
        stack=stack,
        update_count=5,
    )

    assert learner.public_heuristic_coef == pytest.approx(0.5)
    assert learner.teacher_aux_coefs["family"] == pytest.approx(0.01875)
    assert learner.teacher_aux_coefs["slot"] == pytest.approx(0.05)
    assert learner.teacher_aux_coefs["hand"] == pytest.approx(0.04375)
    assert learner.teacher_aux_coefs["action"] == pytest.approx(0.03125)
    assert learner.teacher_aux_coefs["same_family_action"] == pytest.approx(0.075)
    assert learner.teacher_aux_coefs["action_margin"] == pytest.approx(0.0625)
    assert model.learner_scale == pytest.approx(0.5)
    assert model.actor_scale == pytest.approx(0.4)
    assert metrics == {
        "teacher_public_heuristic_coef_active": pytest.approx(0.5),
        "teacher_supervised_coef_scale_active": pytest.approx(0.625),
        "teacher_family_coef_active": pytest.approx(0.01875),
        "teacher_slot_coef_active": pytest.approx(0.05),
        "teacher_hand_coef_active": pytest.approx(0.04375),
        "teacher_move_source_coef_active": pytest.approx(0.0125),
        "teacher_attack_type_coef_active": pytest.approx(0.025),
        "teacher_action_coef_active": pytest.approx(0.03125),
        "teacher_same_family_action_coef_active": pytest.approx(0.075),
        "teacher_action_margin_coef_active": pytest.approx(0.0625),
        "teacher_same_family_action_margin_coef_active": pytest.approx(0.0375),
        "teacher_label_profile_id_active": pytest.approx(0.0),
        "teacher_label_profile_base_active": pytest.approx(1.0),
        "teacher_label_profile_aggressive_active": pytest.approx(0.0),
        "teacher_label_profile_control_active": pytest.approx(0.0),
        "public_heuristic_logit_bias_scale_active": pytest.approx(0.5),
        "public_heuristic_actor_logit_bias_scale_active": pytest.approx(0.4),
    }


def test_model_guidance_payload_round_trips_missing_learner_scale_from_current_model() -> None:
    source = _FakeGuidedModel(learner_scale=0.25, actor_scale=0.75)
    assert model_guidance_payload(source) == {
        "public_heuristic_logit_bias_scale": pytest.approx(0.25),
        "public_heuristic_actor_logit_bias_scale": pytest.approx(0.75),
    }

    target = _FakeGuidedModel(learner_scale=0.6, actor_scale=0.1)
    restore_model_guidance_from_payload(
        target,
        {"public_heuristic_actor_logit_bias_scale": 0.35},
    )

    assert target.learner_scale == pytest.approx(0.6)
    assert target.actor_scale == pytest.approx(0.35)
