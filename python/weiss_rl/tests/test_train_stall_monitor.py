from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import scripts.train as train_script
import torch
from scripts.train import (
    MinimalRollout,
    PeriodicDevEvalOpponentSpec,
    TrainingPaths,
    _build_learner_batch,
    _checkpoint_candidate_metric,
    _confirmatory_dev_eval_request,
    _dev_eval_ineligibility_reasons,
    _entropy_coef_for_next_update,
    _expand_periodic_dev_eval_paired_seeds,
    _periodic_dev_eval_opponents,
    _pin_snapshot_ids,
    _resolve_async_periodic_dev_eval_device,
    _resolve_periodic_dev_eval_opponent_specs,
    _resolved_periodic_dev_eval_worker_devices,
    _should_promote_best_checkpoint,
    _unpin_snapshot_ids,
    _update_early_cutoff,
    _update_stall_monitor,
)

from weiss_rl.config import apply_stack_overrides, load_stack_config
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _training_paths(tmp_path: Path) -> TrainingPaths:
    training_dir = tmp_path / "training"
    logs_dir = training_dir / "logs"
    snapshots_dir = training_dir / "snapshots"
    checkpoints_dir = training_dir / "checkpoints"
    tensorboard_dir = tmp_path / "tensorboard"
    for path in (logs_dir, snapshots_dir, checkpoints_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
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


def test_update_stall_monitor_marks_run_after_consecutive_truncating_evals(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_paths = _training_paths(tmp_path)
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
    training_paths = _training_paths(tmp_path)
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


def test_update_early_cutoff_triggers_after_patience_without_meaningful_improvement(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    stack = apply_stack_overrides(
        stack,
        {
            "curriculum.early_cutoff.enabled": True,
            "curriculum.early_cutoff.warmup_updates": 20,
            "curriculum.early_cutoff.patience_updates": 20,
            "curriculum.early_cutoff.min_improvement": 0.01,
            "curriculum.early_cutoff.stall_patience_evals": 0,
        },
    )
    training_paths = _training_paths(tmp_path)

    first = _update_early_cutoff(
        stack=stack,
        training_paths=training_paths,
        update_count=20,
        summary_payload={"aggregate_score": 0.60},
    )
    second = _update_early_cutoff(
        stack=stack,
        training_paths=training_paths,
        update_count=40,
        summary_payload={"aggregate_score": 0.605},
    )

    assert first is not None
    assert second is not None
    assert first["should_stop"] is False
    assert second["should_stop"] is True
    assert second["reasons"] == ["no_improvement"]
    assert second["best_update_count"] == 20


def test_update_early_cutoff_triggers_after_repeated_stall_evals(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    stack = apply_stack_overrides(
        stack,
        {
            "curriculum.early_cutoff.enabled": True,
            "curriculum.early_cutoff.patience_updates": 0,
            "curriculum.early_cutoff.stall_patience_evals": 2,
            "curriculum.early_cutoff.stall_rate_threshold": 0.25,
        },
    )
    training_paths = _training_paths(tmp_path)
    payload = {
        "aggregate_score": 0.50,
        "anchors": {
            "B0 RandomLegal": {"summary": {"games": 10, "no_progress_timeouts": 4}},
            "B1 NoLeague baseline": {"summary": {"games": 10, "no_progress_timeouts": 1}},
        },
    }

    first = _update_early_cutoff(
        stack=stack,
        training_paths=training_paths,
        update_count=10,
        summary_payload=payload,
    )
    second = _update_early_cutoff(
        stack=stack,
        training_paths=training_paths,
        update_count=20,
        summary_payload=payload,
    )

    assert first is not None
    assert second is not None
    assert first["should_stop"] is False
    assert second["should_stop"] is True
    assert second["reasons"] == ["stall"]
    assert second["consecutive_stall_evals"] == 2


def test_dev_eval_ineligibility_reasons_still_enforces_checkpoint_guard_when_stall_monitor_is_disabled() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    stack = apply_stack_overrides(
        stack,
        {
            "curriculum.stall_monitor.enabled": False,
            "curriculum.checkpoint_guard.enabled": True,
            "curriculum.checkpoint_guard.promote_min_prob_gt_half": 0.6,
            "curriculum.checkpoint_guard.promote_max_ci_half_width": 0.24,
        },
    )

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.70,
            "anchors": {
                "B0 RandomLegal": {
                    "uncertainty": {
                        "prob_gt_half": 0.55,
                        "prob_lt_half": 0.10,
                        "ci_half_width": 0.30,
                    }
                }
            },
        },
    )

    assert reasons == ("confidence_prob", "confidence_ci")


def test_drop_stale_pending_promotion_gate_discards_candidates_newer_than_rollback_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_paths = _training_paths(tmp_path)
    observed: list[tuple[str, ...]] = []

    def _fake_unpin_snapshot_ids(**kwargs: object) -> None:
        observed.append(tuple(cast(tuple[str, ...], kwargs["snapshot_ids"])))

    monkeypatch.setattr(train_script, "_unpin_snapshot_ids", _fake_unpin_snapshot_ids)
    stale_gate = SimpleNamespace(
        request=SimpleNamespace(update_count=220, candidate_policy_id="policy_000220"),
        future=object(),
        pinned_snapshot_ids=("policy_000220", "anchor_000160"),
    )
    retained_gate = SimpleNamespace(
        request=SimpleNamespace(update_count=160, candidate_policy_id="policy_000160"),
        future=object(),
        pinned_snapshot_ids=("policy_000160", "anchor_000160"),
    )

    assert train_script._drop_stale_pending_promotion_gate(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        pending_gate=stale_gate,
        rollback_best_update_count=160,
    ) is None
    assert observed == [("policy_000220", "anchor_000160")]
    assert (
        train_script._drop_stale_pending_promotion_gate(
            stack=stack,
            training_paths=training_paths,
            run_dir=tmp_path,
            pending_gate=retained_gate,
            rollback_best_update_count=160,
        )
        is retained_gate
    )
    assert observed == [("policy_000220", "anchor_000160")]


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
        stack=stack,
        contract=contract,
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
            evaluation=SimpleNamespace(eval_device="cpu"),
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
        stack=stack,
        contract=contract,
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
            evaluation=SimpleNamespace(eval_device="cpu"),
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
        stack=stack,
        contract=contract,
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


def test_resolve_periodic_dev_eval_opponent_specs_returns_explicit_snapshot_specs(tmp_path: Path) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("Previous recent snapshot",),
                )
            )
        )
    )
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
        policy_id="policy_000020",
        update=20,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000020"),
    )
    registry.add_snapshot(
        policy_id="policy_000030",
        update=30,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000030"),
    )
    registry.save(snapshots_dir / "registry.json")

    specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(stack=stack, run_dir=tmp_path)

    assert specs == (
        PeriodicDevEvalOpponentSpec(
            policy_id="b0_randomlegal",
            display_name="B0 RandomLegal",
            kind="random_legal",
            snapshot_path=None,
            heuristic_profile=None,
        ),
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="B1 NoLeague baseline",
            kind="snapshot",
            snapshot_path=snapshot_weights_relpath("b1_noleague_baseline"),
            heuristic_profile=None,
        ),
        PeriodicDevEvalOpponentSpec(
            policy_id="policy_000020",
            display_name="Previous recent snapshot",
            kind="snapshot",
            snapshot_path=snapshot_weights_relpath("policy_000020"),
            heuristic_profile=None,
        ),
    )
    assert pinned_snapshot_ids == ("b1_noleague_baseline", "policy_000020")


def test_pin_and_unpin_snapshot_ids_preserve_existing_pins(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "presets" / "typed_local.yaml")
    training_dir = tmp_path / "training"
    snapshots_dir = training_dir / "snapshots"
    checkpoints_dir = training_dir / "checkpoints"
    logs_dir = training_dir / "logs"
    tensorboard_dir = tmp_path / "tensorboard"
    for path in (snapshots_dir, checkpoints_dir, logs_dir, tensorboard_dir):
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
    registry = SnapshotRegistry(recent_size=4, champion_size=1)
    registry.add_snapshot(
        policy_id="baseline",
        update=1,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("baseline"),
    )
    registry.add_snapshot(
        policy_id="candidate",
        update=2,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("candidate"),
    )
    registry.pin_snapshot("baseline")
    registry.save(snapshots_dir / "registry.json")

    newly_pinned = _pin_snapshot_ids(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        snapshot_ids=("baseline", "candidate"),
    )
    assert newly_pinned == ("candidate",)

    _unpin_snapshot_ids(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        snapshot_ids=newly_pinned,
    )
    reloaded = SnapshotRegistry.load(snapshots_dir / "registry.json")
    assert reloaded.pinned_snapshots == ["baseline"]


def test_resolve_async_periodic_dev_eval_device_prefers_non_learner_actor_gpu(monkeypatch) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(eval_device="cuda:auto"),
            system=SimpleNamespace(actor_process_count=4),
        )
    )
    monkeypatch.setattr(
        train_script,
        "resolve_actor_device_layout",
        lambda stack, actor_count, learner_device, prefer_process_collectors: ("cuda:1", "cuda:2"),
    )

    resolved = _resolve_async_periodic_dev_eval_device(
        stack=stack,
        learner_device=torch.device("cuda:0"),
    )

    assert resolved == "cuda:2"


def test_resolved_periodic_dev_eval_worker_devices_cycles_explicit_devices() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(system=None))

    resolved = _resolved_periodic_dev_eval_worker_devices(
        stack=stack,
        parallel_workers=4,
        explicit_worker_devices=("cuda:0", "cuda:2"),
        eval_device="cuda:auto",
        learner_device=None,
    )

    assert resolved == ("cuda:0", "cuda:2", "cuda:0", "cuda:2")


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
