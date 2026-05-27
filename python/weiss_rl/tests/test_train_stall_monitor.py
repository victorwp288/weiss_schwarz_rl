from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import scripts.train as train_script
import torch
from scripts.train import (
    MinimalRollout,
    TrainingPaths,
    _build_learner_batch,
    _checkpoint_candidate_metric,
    _confirmatory_dev_eval_request,
    _dev_eval_ineligibility_reasons,
    _entropy_coef_for_next_update,
    _expand_periodic_dev_eval_paired_seeds,
    _periodic_dev_eval_opponents,
    _should_promote_best_checkpoint,
    _update_stall_monitor,
)

from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath


def _repo_root():
    return Path(__file__).resolve().parents[3]


def test_update_stall_monitor_marks_run_after_consecutive_truncating_evals(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_dir = tmp_path / "training"
    logs_dir = training_dir / "logs"
    snapshots_dir = training_dir / "snapshots"
    checkpoints_dir = training_dir / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    for path in (logs_dir, snapshots_dir, checkpoints_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    training_paths = TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=tensorboard_dir,
        scalars_path=logs_dir / "training_metrics.jsonl",
        performance_log_path=logs_dir / "performance.jsonl",
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        checkpoint_tracker_path=checkpoints_dir / "checkpoint_tracker.json",
    )
    payload = {
        "anchors": {
            "B0 RandomLegal": {"summary": {"games": 10, "truncations": 4}},
            "B1 NoLeague baseline": {"summary": {"games": 10, "truncations": 3}},
        }
    }

    first = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=100,
        summary_payload=payload,
    )
    second = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=200,
        summary_payload=payload,
    )

    assert first is not None
    assert second is not None
    assert first["stall_risk"] is False
    assert second["stall_risk"] is True
    assert second["worst_anchor"] == "B0 RandomLegal"


def test_update_stall_monitor_includes_optional_b2_anchor(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_dir = tmp_path / "training"
    logs_dir = training_dir / "logs"
    snapshots_dir = training_dir / "snapshots"
    checkpoints_dir = training_dir / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    for path in (logs_dir, snapshots_dir, checkpoints_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    training_paths = TrainingPaths(
        training_dir=training_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        snapshots_dir=snapshots_dir,
        tensorboard_dir=tensorboard_dir,
        scalars_path=logs_dir / "training_metrics.jsonl",
        performance_log_path=logs_dir / "performance.jsonl",
        latest_checkpoint_path=checkpoints_dir / "latest.pt",
        best_checkpoint_path=checkpoints_dir / "best.pt",
        checkpoint_tracker_path=checkpoints_dir / "checkpoint_tracker.json",
    )
    payload = {
        "anchors": {
            "B0 RandomLegal": {"summary": {"games": 10, "truncations": 1}},
            "B1 NoLeague baseline": {"summary": {"games": 10, "truncations": 2}},
            "B2 HeuristicPublic": {"summary": {"games": 10, "truncations": 9}},
        }
    }

    first = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=100,
        summary_payload=payload,
    )
    second = _update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=200,
        summary_payload=payload,
    )

    assert first is not None
    assert second is not None
    assert second["stall_risk"] is True
    assert second["worst_anchor"] == "B2 HeuristicPublic"


def test_train_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = MinimalRollout(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_mask=np.ones((2, 1, 2), dtype=np.bool_),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        logits=np.zeros((2, 1, 2), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
    )

    batch = _build_learner_batch(
        stack,
        rollout,
        np.zeros((1,), dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])


def test_checkpoint_candidate_metric_prefers_aggregate_score() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 1.0},
        dev_eval_summary={
            "aggregate_score": 0.625,
            "uncertainty": {"mean": 0.125},
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert metric_kind == "dev_eval_mean"
    assert metric_value == pytest.approx(0.625)


def test_checkpoint_candidate_metric_rejects_truncation_heavy_dev_eval() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.75},
        dev_eval_summary={
            "aggregate_score": 0.8,
            "stall_monitor": {"worst_truncation_rate": 0.5},
        },
    )

    assert metric_kind is None
    assert metric_value is None


def test_checkpoint_candidate_metric_rejects_low_confidence_dev_eval() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={
            "aggregate_score": 0.8,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 0.95, "prob_lt_half": 0.05, "ci_half_width": 0.1}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.52, "prob_lt_half": 0.48, "ci_half_width": 0.28}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert metric_kind is None
    assert metric_value is None


def test_checkpoint_candidate_metric_waits_for_dev_eval_when_periodic_eval_enabled() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    metric_kind, metric_value = _checkpoint_candidate_metric(
        stack=stack,
        latest_metrics={"loss": 0.5},
        dev_eval_summary=None,
    )

    assert metric_kind is None
    assert metric_value is None


def test_should_not_promote_best_checkpoint_when_candidate_metric_is_missing() -> None:
    assert (
        _should_promote_best_checkpoint(
            existing_record=None,
            candidate_kind=None,
            candidate_value=None,
        )
        is False
    )


def test_periodic_dev_eval_opponents_include_optional_b2_when_available(monkeypatch, tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    fake_heuristic = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=2, champion_size=1)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    class _FakeHeuristicPolicy:
        @staticmethod
        def from_spec_bundle(spec_bundle):
            return fake_heuristic

    monkeypatch.setattr(train_script, "HeuristicPublicPolicy", _FakeHeuristicPolicy)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[0] for item in opponents] == [
        "b0_randomlegal",
        "b1_noleague_baseline",
        "B2 HeuristicPublic",
    ]
    assert opponents[1][2] is fake_model
    assert opponents[2][3] is fake_heuristic


def test_periodic_dev_eval_opponents_include_extra_snapshot_anchor_from_promotion_anchor_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("policy_000123",),
                )
            )
        )
    )
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=4, champion_size=1)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.add_snapshot(
        policy_id="policy_000123",
        update=123,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000123"),
    )
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[1] for item in opponents] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "policy_000123",
    ]
    assert opponents[1][2] is fake_model
    assert opponents[2][0] == "policy_000123"
    assert opponents[2][2] is fake_model


def test_periodic_dev_eval_opponents_resolve_symbolic_snapshot_anchor_aliases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("Previous champion snapshot", "Previous recent snapshot"),
                )
            )
        )
    )
    contract = SimpleNamespace(spec_bundle={"observation": {"kind": "stub"}, "action": {"kind": "stub"}})
    fake_model = object()
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=4, champion_size=2)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.add_snapshot(
        policy_id="policy_000010",
        update=10,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000010"),
    )
    registry.add_snapshot(
        policy_id="policy_000020",
        update=20,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000020"),
    )
    registry.add_snapshot(
        policy_id="policy_000030",
        update=30,
        weights_sha256="d" * 64,
        path=snapshot_weights_relpath("policy_000030"),
    )
    registry.add_champion("policy_000010")
    registry.add_champion("policy_000020")
    registry.save(snapshots_dir / "registry.json")

    monkeypatch.setattr(train_script, "_load_snapshot_eval_model", lambda **kwargs: fake_model)

    opponents = _periodic_dev_eval_opponents(
        stack=cast(Any, stack),
        contract=cast(Any, contract),
        run_dir=tmp_path,
        observation_dim=1,
        action_dim=2,
    )

    assert [item[1] for item in opponents] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "Previous champion snapshot",
        "Previous recent snapshot",
    ]
    assert opponents[2][0] == "policy_000010"
    assert opponents[2][2] is fake_model
    assert opponents[3][0] == "policy_000020"
    assert opponents[3][2] is fake_model


def test_dev_eval_ineligibility_reasons_identify_borderline_confidence_only() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.625,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.1467}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.686, "prob_lt_half": 0.314, "ci_half_width": 0.2492}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert reasons == ("confidence_ci",)


def test_dev_eval_ineligibility_reasons_apply_checkpoint_confidence_when_stall_monitor_disabled() -> None:
    stack = load_stack_config(
        _repo_root()
        / "configs/thesis/ablations/public_teacher_b2exact_filteredexact_constpublic_antipass02_lowentropy_choiceexactmargin_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )
    assert stack.config.curriculum is not None
    assert stack.config.curriculum.stall_monitor.enabled is False
    assert stack.config.curriculum.checkpoint_guard.enabled is True

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.640625,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.07}},
                "B2 HeuristicPublic": {
                    "uncertainty": {"prob_gt_half": 0.021, "prob_lt_half": 0.979, "ci_half_width": 0.16}
                },
            },
        },
    )

    assert reasons == ("confidence_prob",)


def test_confirmatory_dev_eval_request_targets_borderline_score_drop_for_reevaluation() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.84375},
        dev_eval_summary={
            "aggregate_score": 0.71875,
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is not None
    assert request["reasons"] == ["score_drop"]
    assert request["current_score"] == pytest.approx(0.71875)
    assert request["existing_best_score"] == pytest.approx(0.84375)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_targets_score_improving_borderline_candidate() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.59375},
        dev_eval_summary={
            "aggregate_score": 0.625,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.1467}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.686, "prob_lt_half": 0.314, "ci_half_width": 0.2492}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is not None
    assert request["reasons"] == ["confidence_ci"]
    assert request["current_score"] == pytest.approx(0.625)
    assert request["existing_best_score"] == pytest.approx(0.59375)
    assert request["ci_excess"] == pytest.approx(0.0092, abs=1e-4)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_targets_multianchor_near_miss_candidate() -> None:
    stack = load_stack_config(
        _repo_root() / "configs/thesis/ablations/"
        "public_teacher_b2exact_margin_multianchor_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=None,
        dev_eval_summary={
            "aggregate_score": 0.6953125,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"mean": 1.0, "prob_gt_half": 1.0, "ci_half_width": 0.0}},
                "B2 HeuristicPublic": {"uncertainty": {"mean": 0.6875, "prob_gt_half": 0.989, "ci_half_width": 0.14}},
                "B3 HeuristicPublicAggro": {
                    "uncertainty": {"mean": 0.625, "prob_gt_half": 0.899, "ci_half_width": 0.17}
                },
                "B4 HeuristicPublicControl": {
                    "uncertainty": {"mean": 0.46875, "prob_gt_half": 0.359, "ci_half_width": 0.18}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is not None
    assert request["reasons"] == ["confidence_prob"]
    assert request["current_score"] == pytest.approx(0.6953125)
    assert request["worst_anchor_mean"] == pytest.approx(0.46875)
    assert int(request["target_pairs"]) >= 32


def test_confirmatory_dev_eval_request_rejects_multianchor_clear_anchor_failure() -> None:
    stack = load_stack_config(
        _repo_root() / "configs/thesis/ablations/"
        "public_teacher_b2exact_margin_multianchor_argmaxdev_attackguard_mainmoveguard_mulliganguard_reward.yaml"
    )

    request = _confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=None,
        dev_eval_summary={
            "aggregate_score": 0.6171875,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"mean": 1.0, "prob_gt_half": 1.0, "ci_half_width": 0.0}},
                "B2 HeuristicPublic": {"uncertainty": {"mean": 0.65625, "prob_gt_half": 0.962, "ci_half_width": 0.16}},
                "B3 HeuristicPublicAggro": {
                    "uncertainty": {"mean": 0.4375, "prob_gt_half": 0.212, "ci_half_width": 0.17}
                },
                "B4 HeuristicPublicControl": {
                    "uncertainty": {"mean": 0.375, "prob_gt_half": 0.03, "ci_half_width": 0.13}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert request is None


def test_expand_periodic_dev_eval_paired_seeds_is_deterministic_and_unique() -> None:
    base_paired_seeds = list(range(8))

    expanded_a = _expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=32,
        seed_file_sha256="abc123",
        update_count=200,
        policy_version=10,
        scope="periodic_dev_eval_confirmatory",
    )
    expanded_b = _expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=32,
        seed_file_sha256="abc123",
        update_count=200,
        policy_version=10,
        scope="periodic_dev_eval_confirmatory",
    )

    assert expanded_a[:8] == base_paired_seeds
    assert expanded_a == expanded_b
    assert len(expanded_a) == 32
    assert len(set(expanded_a)) == 32


def test_entropy_coef_for_next_update_linearly_anneals() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training = stack.config.training
    assert training is not None

    assert _entropy_coef_for_next_update(training, update_count=0) == pytest.approx(training.entropy_coef)
    midpoint = _entropy_coef_for_next_update(
        training,
        update_count=int(training.entropy_anneal_steps_updates // 2),
    )
    assert midpoint == pytest.approx((training.entropy_coef + training.entropy_anneal_to) / 2.0)
    assert _entropy_coef_for_next_update(
        training,
        update_count=int(training.entropy_anneal_steps_updates * 2),
    ) == pytest.approx(training.entropy_anneal_to)
