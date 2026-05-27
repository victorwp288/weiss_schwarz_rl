from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from torch import nn

from weiss_rl.training.learner_factory import build_training_learner


class _TinyModel(nn.Module):
    pass


class _CapturedImpalaLearner:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _CapturedPpoLearner:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _training_config(*, trajectory_retention_coef: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        learning_rate=0.01,
        value_loss_coef=0.5,
        entropy_coef=0.02,
        entropy_scope="family",
        grad_norm_clip=3.0,
        mixed_precision=1,
        teacher_family_coef=0.1,
        teacher_slot_coef=0.2,
        teacher_hand_coef=0.25,
        teacher_move_source_coef=0.3,
        teacher_attack_type_coef=0.4,
        teacher_action_coef=0.5,
        teacher_same_family_action_coef=0.6,
        teacher_action_margin_coef=0.65,
        teacher_action_margin=0.75,
        teacher_same_family_action_margin_coef=0.35,
        teacher_same_family_action_margin=0.45,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=0.7,
        teacher_public_heuristic_temperature=1.5,
        teacher_public_nonpass_over_pass_coef=0.08,
        teacher_public_nonpass_over_pass_margin=0.25,
        teacher_public_heuristic_families=("play", "attack"),
        teacher_public_heuristic_profiles=("base", "tempo"),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=8,
        policy_anchor_coef=0.09,
        policy_anchor_top_action_coef=0.04,
        policy_anchor_temperature=0.8,
        trajectory_retention_coef=trajectory_retention_coef,
        profile_timers=True,
        structured_metrics_mode="compact",
        teacher_aux_mode="warmstart_only",
        vtrace_rho_bar=1.25,
        vtrace_c_bar=0.75,
        ppo_clip_epsilon=0.11,
        ppo_value_clip_epsilon=0.22,
        ppo_epochs=4,
        ppo_target_kl=0.33,
        ppo_normalize_advantages=0,
    )


def _training_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints", logs_dir=tmp_path / "logs")


def test_build_training_learner_passes_common_and_impala_specific_kwargs(tmp_path: Path) -> None:
    learner = build_training_learner(
        algorithm="impala_vtrace_structured_v1",
        model=_TinyModel(),
        compiled_model=None,
        training_config=_training_config(trajectory_retention_coef=0.12),
        training_paths=_training_paths(tmp_path),
        pass_action_id=17,
        checkpoint_interval_updates=9,
        impala_learner_cls=_CapturedImpalaLearner,
    )

    kwargs = cast(Any, learner).kwargs
    assert kwargs["learning_rate"] == 0.01
    assert kwargs["mixed_precision"] is True
    assert kwargs["checkpoint_dir"] == tmp_path / "checkpoints"
    assert kwargs["logs_dir"] == tmp_path / "logs"
    assert kwargs["pass_action_id"] == 17
    assert kwargs["checkpoint_interval_updates"] == 9
    assert kwargs["logging_interval_updates"] == 1
    assert kwargs["teacher_public_heuristic_profiles"] == ("base", "tempo")
    assert kwargs["teacher_hand_coef"] == 0.25
    assert kwargs["teacher_public_heuristic_profile_mode"] == "cycle"
    assert kwargs["teacher_action_margin_coef"] == 0.65
    assert kwargs["teacher_action_margin"] == 0.75
    assert kwargs["teacher_same_family_action_margin_coef"] == 0.35
    assert kwargs["teacher_same_family_action_margin"] == 0.45
    assert kwargs["teacher_exact_action_families"] == ("attack",)
    assert kwargs["teacher_public_nonpass_over_pass_coef"] == 0.08
    assert kwargs["teacher_public_nonpass_over_pass_margin"] == 0.25
    assert kwargs["policy_anchor_coef"] == 0.09
    assert kwargs["policy_anchor_top_action_coef"] == 0.04
    assert kwargs["policy_anchor_temperature"] == 0.8
    assert kwargs["trajectory_retention_coef"] == 0.12
    assert kwargs["profile_timers"] is True
    assert kwargs["structured_metrics_mode"] == "compact"
    assert kwargs["teacher_aux_mode"] == "warmstart_only"
    assert kwargs["vtrace_rho_bar"] == 1.25
    assert kwargs["vtrace_c_bar"] == 0.75
    assert kwargs["entropy_scope"] == "family"


def test_build_training_learner_passes_ppo_specific_kwargs(tmp_path: Path) -> None:
    learner = build_training_learner(
        algorithm="ppo_lite_masked_v1",
        model=_TinyModel(),
        compiled_model=None,
        training_config=_training_config(),
        training_paths=_training_paths(tmp_path),
        pass_action_id=3,
        checkpoint_interval_updates=5,
        ppo_lite_learner_cls=_CapturedPpoLearner,
    )

    kwargs = cast(Any, learner).kwargs
    assert kwargs["ppo_clip_epsilon"] == 0.11
    assert kwargs["value_clip_epsilon"] == 0.22
    assert kwargs["ppo_epochs"] == 4
    assert kwargs["target_kl"] == 0.33
    assert kwargs["normalize_advantages"] is False


def test_build_training_learner_rejects_ppo_trajectory_retention_noop(tmp_path: Path) -> None:
    try:
        build_training_learner(
            algorithm="ppo_lite_masked_v1",
            model=_TinyModel(),
            compiled_model=None,
            training_config=_training_config(trajectory_retention_coef=0.1),
            training_paths=_training_paths(tmp_path),
            pass_action_id=3,
            checkpoint_interval_updates=5,
            ppo_lite_learner_cls=_CapturedPpoLearner,
        )
    except RuntimeError as exc:
        assert str(exc) == "training.structured_aux.trajectory_retention_coef is only supported by IMPALA/V-trace"
    else:
        raise AssertionError("PPO retention should fail loudly instead of silently doing nothing")


def test_build_training_learner_rejects_unknown_algorithm(tmp_path: Path) -> None:
    try:
        build_training_learner(
            algorithm="not_real",
            model=_TinyModel(),
            compiled_model=None,
            training_config=_training_config(),
            training_paths=_training_paths(tmp_path),
            pass_action_id=3,
            checkpoint_interval_updates=5,
        )
    except RuntimeError as exc:
        assert str(exc) == "Unsupported training.algorithm: not_real"
    else:
        raise AssertionError("unknown algorithms must fail loudly")
