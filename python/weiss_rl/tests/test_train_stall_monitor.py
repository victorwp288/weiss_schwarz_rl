from __future__ import annotations

import json
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import scripts.train as train_script
import torch
import weiss_rl.eval.simulator_runner as simulator_runner
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
    _league_eval_warmup_gate_status,
    _periodic_dev_eval_opponents,
    _pin_snapshot_ids,
    _reference_policy_top_action_bc_coef_for_next_update,
    _reference_policy_top_action_family_bc_coef_for_next_update,
    _resolve_async_periodic_dev_eval_device,
    _resolve_periodic_dev_eval_opponent_specs,
    _resolved_periodic_dev_eval_worker_devices,
    _should_promote_best_checkpoint,
    _unpin_snapshot_ids,
    _update_early_cutoff,
    _update_stall_monitor,
    _weighted_dev_eval_aggregate,
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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


def test_failed_async_periodic_dev_eval_unpins_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    training_paths = _training_paths(tmp_path)
    future: Future[dict[str, object]] = Future()
    future.set_exception(RuntimeError("worker failed"))
    pending_eval = train_script.PendingPeriodicDevEval(
        future=future,
        request=SimpleNamespace(),
        pinned_snapshot_ids=("anchor_a", "anchor_b"),
        latest_metrics={},
    )
    observed: list[tuple[str, ...]] = []

    def _fake_unpin_snapshot_ids(**kwargs: object) -> None:
        observed.append(tuple(cast(tuple[str, ...], kwargs["snapshot_ids"])))

    monkeypatch.setattr(train_script, "_unpin_snapshot_ids", _fake_unpin_snapshot_ids)

    with pytest.raises(RuntimeError, match="worker failed"):
        train_script._process_completed_periodic_dev_eval(
            pending_eval=pending_eval,
            stack=stack,
            contract=SimpleNamespace(),
            artifacts=SimpleNamespace(run_dir=tmp_path),
            training_paths=training_paths,
            runtime=SimpleNamespace(),
            learner=SimpleNamespace(),
            device=torch.device("cpu"),
            run_id256="ab" * 32,
            config_hash256="cd" * 32,
            spec_hash256="ef" * 32,
            last_rollback_update=None,
            tensorboard_logger=None,
        )

    assert observed == [("anchor_a", "anchor_b")]


def test_failed_async_promotion_gate_unpins_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    training_paths = _training_paths(tmp_path)
    future: Future[dict[str, object]] = Future()
    future.set_exception(RuntimeError("gate failed"))
    pending_gate = train_script.PendingPromotionGate(
        future=future,
        request=SimpleNamespace(),
        pinned_snapshot_ids=("candidate", "anchor"),
    )
    observed: list[tuple[str, ...]] = []

    def _fake_unpin_snapshot_ids(**kwargs: object) -> None:
        observed.append(tuple(cast(tuple[str, ...], kwargs["snapshot_ids"])))

    monkeypatch.setattr(train_script, "_unpin_snapshot_ids", _fake_unpin_snapshot_ids)

    with pytest.raises(RuntimeError, match="gate failed"):
        train_script._process_completed_promotion_gate(
            pending_gate=pending_gate,
            stack=stack,
            artifacts=SimpleNamespace(run_dir=tmp_path),
            training_paths=training_paths,
        )

    assert observed == [("candidate", "anchor")]


def test_failed_async_promotion_gate_marks_candidate_rejected(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    training_paths = _training_paths(tmp_path)
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="candidate",
        update=10,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("candidate"),
    )
    registry.save(training_paths.snapshots_dir / "registry.json")
    future: Future[dict[str, object]] = Future()
    future.set_result(
        {
            "passed": False,
            "candidate_policy_id": "candidate",
            "update_count": 10,
            "policy_version": 10,
            "ordered_opponents": [],
            "reasons": [{"code": "low_score"}],
        }
    )
    pending_gate = train_script.PendingPromotionGate(
        future=future,
        request=SimpleNamespace(),
        pinned_snapshot_ids=(),
    )

    promoted = train_script._process_completed_promotion_gate(
        pending_gate=pending_gate,
        stack=stack,
        artifacts=SimpleNamespace(run_dir=tmp_path),
        training_paths=training_paths,
    )

    assert promoted is False
    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert reloaded.rejected_snapshots == ["candidate"]


def test_periodic_dev_eval_single_worker_honors_worker_device_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    observed: dict[str, object] = {}
    seed_file = tmp_path / "dev_eval_seeds.txt"

    def _fake_periodic_dev_eval_schedule(_stack: object):
        return seed_file, {}, [123], "ab" * 32

    def _fake_run_periodic_dev_eval_matchups_for_opponents(**kwargs: object) -> list[dict[str, object]]:
        observed["eval_device_override"] = kwargs["eval_device_override"]
        return [
            {
                "display_name": "B0 RandomLegal",
                "policy_id": "b0_randomlegal",
                "matchup_payload": {
                    "uncertainty": {"mean": 0.5},
                    "games": 2,
                    "truncations": 0,
                },
            }
        ]

    monkeypatch.setattr(train_script, "_periodic_dev_eval_schedule", _fake_periodic_dev_eval_schedule)
    monkeypatch.setattr(
        train_script,
        "_run_periodic_dev_eval_matchups_for_opponents",
        _fake_run_periodic_dev_eval_matchups_for_opponents,
    )

    summary = train_script._run_periodic_dev_eval_for_checkpoint(
        stack=stack,
        contract=SimpleNamespace(
            spec_bundle={
                "observation": {"obs_len": 4},
                "action": {"action_space_size": 3},
            }
        ),
        run_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        focal_policy_id="policy_000010",
        update_count=10,
        policy_version=10,
        run_id256="12" * 32,
        config_hash256="34" * 32,
        spec_hash256="56" * 32,
        opponent_specs=(
            PeriodicDevEvalOpponentSpec(
                policy_id="b0_randomlegal",
                display_name="B0 RandomLegal",
                kind="random_legal",
            ),
        ),
        eval_device_override="cuda:auto",
        parallel_workers_override=1,
        parallel_worker_devices_override=("cpu",),
    )

    assert observed["eval_device_override"] == "cpu"
    assert summary["periodic_dev_eval_parallel"] == {
        "batched_inference_enabled": False,
        "enabled": False,
        "job_count": 1,
        "seed_block_sharding_enabled": False,
        "worker_count": 1,
        "worker_devices": ["cpu"],
    }


def test_periodic_dev_eval_seed_block_jobs_expand_beyond_anchor_count() -> None:
    opponents = (
        PeriodicDevEvalOpponentSpec(policy_id="a", display_name="A", kind="random_legal"),
        PeriodicDevEvalOpponentSpec(policy_id="b", display_name="B", kind="random_legal"),
    )

    jobs = train_script._build_periodic_dev_eval_seed_block_jobs(
        opponent_specs=opponents,
        paired_seeds=(11, 22, 33, 44),
        configured_parallel_workers=6,
    )

    assert len(jobs) == 6
    assert [job.block_index for job in jobs[:3]] == [0, 1, 2]
    assert jobs[0].paired_seed_items == ((0, 11), (1, 22))
    assert jobs[1].paired_seed_items == ((2, 33),)
    assert jobs[2].paired_seed_items == ((3, 44),)
    assert [job.opponent_index for job in jobs] == [0, 0, 0, 1, 1, 1]


def test_promotion_gate_seed_block_jobs_expand_beyond_anchor_count() -> None:
    anchors = (
        PeriodicDevEvalOpponentSpec(policy_id="b0_randomlegal", display_name="B0 RandomLegal", kind="random_legal"),
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="B1 NoLeague baseline",
            kind="random_legal",
        ),
    )

    jobs = train_script._build_promotion_gate_seed_block_jobs(
        anchor_specs=anchors,
        paired_seeds=(11, 22, 33, 44),
        configured_parallel_workers=6,
    )

    assert len(jobs) == 6
    assert [job.block_index for job in jobs[:3]] == [0, 1, 2]
    assert jobs[0].paired_seed_items == ((0, 11), (1, 22))
    assert jobs[1].paired_seed_items == ((2, 33),)
    assert jobs[2].paired_seed_items == ((3, 44),)
    assert [job.anchor_index for job in jobs] == [0, 0, 0, 1, 1, 1]


def test_eval_snapshot_model_cache_eviction_is_lru(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEvalModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1
            return self

    cache = train_script._EVAL_SNAPSHOT_MODEL_CACHE
    cache.clear()
    monkeypatch.setattr(train_script, "_EVAL_SNAPSHOT_MODEL_CACHE_MAX_ENTRIES", 2)
    try:
        model_a = FakeEvalModel("a")
        model_b = FakeEvalModel("b")
        model_c = FakeEvalModel("c")
        train_script._remember_eval_snapshot_model(("a",), model_a)
        train_script._remember_eval_snapshot_model(("b",), model_b)

        assert list(cache.keys()) == [("a",), ("b",)]
        assert train_script._get_cached_eval_snapshot_model(("a",)) is model_a
        assert model_a.eval_calls == 1
        assert list(cache.keys()) == [("b",), ("a",)]

        train_script._remember_eval_snapshot_model(("c",), model_c)

        assert list(cache.keys()) == [("a",), ("c",)]
        assert train_script._get_cached_eval_snapshot_model(("b",)) is None
    finally:
        cache.clear()


def test_parallel_promotion_gate_assembles_seed_block_records_parent_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    assert stack.config.league is not None
    gate = replace(stack.config.league.promotion.gate, parallel_workers=4)
    promotion = replace(stack.config.league.promotion, gate=gate)
    league = replace(stack.config.league, promotion=promotion)
    stack = replace(stack, config=replace(stack.config, league=league))
    anchors = (
        PeriodicDevEvalOpponentSpec(policy_id="a", display_name="A", kind="random_legal"),
        PeriodicDevEvalOpponentSpec(policy_id="b", display_name="B", kind="random_legal"),
    )
    submitted_job_counts: list[int] = []

    def _fake_record(*, pair_index: int, swap_index: int, seed: int, opponent_policy_id: str):
        focal_policy_id = "candidate"
        if swap_index == 0:
            seat0_policy_id = focal_policy_id
            seat1_policy_id = opponent_policy_id
            focal_seat = 0
        else:
            seat0_policy_id = opponent_policy_id
            seat1_policy_id = focal_policy_id
            focal_seat = 1
        return train_script.EvalGameRecord(
            pair_index=pair_index,
            swap_index=swap_index,
            episode_index=pair_index * 2 + swap_index,
            episode_seed=seed,
            episode_key=f"{pair_index * 2 + swap_index + 1:064x}",
            episode_key64=pair_index * 2 + swap_index + 1,
            config_hash256="ab" * 32,
            spec_hash256="cd" * 32,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
            seat0_policy_id=seat0_policy_id,
            seat1_policy_id=seat1_policy_id,
            focal_seat=focal_seat,
            outcome="W",
            terminated=True,
            truncated=False,
            engine_status=0,
            run_id256="12" * 32,
        )

    def _fake_worker(**kwargs: object) -> list[dict[str, object]]:
        jobs = cast(tuple[train_script.PromotionGateSeedBlockJob, ...], kwargs["seed_block_jobs"])
        submitted_job_counts.append(len(jobs))
        payloads: list[dict[str, object]] = []
        for job in jobs:
            records = tuple(
                _fake_record(
                    pair_index=pair_index,
                    swap_index=swap_index,
                    seed=seed,
                    opponent_policy_id=job.anchor_spec.policy_id,
                )
                for pair_index, seed in job.paired_seed_items
                for swap_index in (0, 1)
            )
            payloads.append(
                {
                    "anchor_index": job.anchor_index,
                    "block_index": job.block_index,
                    "anchor_policy_id": job.anchor_spec.policy_id,
                    "anchor_display_name": job.anchor_spec.display_name,
                    "paired_seed_items": job.paired_seed_items,
                    "records": records,
                }
            )
        return payloads

    class FakeExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, fn, **kwargs: object):
            future: Future[list[dict[str, object]]] = Future()
            future.set_result(fn(**kwargs))
            return future

    monkeypatch.setattr(train_script, "_run_parallel_promotion_gate_anchor_worker", _fake_worker)
    monkeypatch.setattr(train_script, "ProcessPoolExecutor", FakeExecutor)

    result = train_script._run_parallel_snapshot_promotion_gate(
        stack=stack,
        artifacts=SimpleNamespace(run_dir=tmp_path),
        training_paths=_training_paths(tmp_path),
        candidate_policy_id="candidate",
        update_count=1,
        policy_version=1,
        run_id256="12" * 32,
        config_hash256="ab" * 32,
        spec_hash256="cd" * 32,
        anchor_policy_ids={"B0 RandomLegal": "b0_randomlegal", "B1 NoLeague baseline": "b1_noleague_baseline"},
        anchor_specs=anchors,
        candidate_snapshot_path="training/snapshots/candidate.pt",
    )

    assert sum(submitted_job_counts) == 4
    assert result.ordered_opponents == ("B0 RandomLegal", "B1 NoLeague baseline")
    assert result.passed is True
    episodes_a = (
        tmp_path / "eval" / "promotion_gate" / "update_1" / "promotion_gate_episodes" / "00_b0_randomlegal.jsonl"
    )
    episodes_b = (
        tmp_path
        / "eval"
        / "promotion_gate"
        / "update_1"
        / "promotion_gate_episodes"
        / "01_b1_noleague_baseline.jsonl"
    )
    assert len(episodes_a.read_text(encoding="utf-8").splitlines()) == 16
    assert len(episodes_b.read_text(encoding="utf-8").splitlines()) == 16


def test_periodic_dev_eval_duplicate_policy_ids_get_unique_matchup_dirs(tmp_path: Path) -> None:
    specs = (
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="B1 NoLeague baseline",
            kind="snapshot",
            snapshot_path="snapshots/b1/weights.pt",
        ),
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="Previous recent snapshot",
            kind="snapshot",
            snapshot_path="snapshots/recent/weights.pt",
        ),
    )

    duplicate_policy_ids = train_script._periodic_dev_eval_duplicate_policy_ids(specs)
    dirs = [
        train_script._periodic_dev_eval_matchup_dir(
            update_dir=tmp_path,
            opponent_spec=spec,
            duplicate_policy_ids=duplicate_policy_ids,
        )
        for spec in specs
    ]

    assert duplicate_policy_ids == {"b1_noleague_baseline"}
    assert dirs[0] != dirs[1]
    assert all(path.name.startswith("b1_noleague_baseline__") for path in dirs)


def test_train_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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


def test_weighted_dev_eval_aggregate_prioritizes_frontier_anchors() -> None:
    score, weights, total_weight = _weighted_dev_eval_aggregate(
        {
            "B0 RandomLegal": 1.0,
            "B1 NoLeague baseline": 0.25,
            "B3 HeuristicPublicAggro": 0.75,
            "Previous recent snapshot": 0.5,
        },
        anchor_weights={
            "B0 RandomLegal": 0.25,
            "B1 NoLeague baseline": 3.0,
            "B3 HeuristicPublicAggro": 2.0,
            "Previous recent snapshot": 2.0,
        },
    )

    assert weights["B1 NoLeague baseline"] == pytest.approx(3.0)
    assert total_weight == pytest.approx(7.25)
    assert score == pytest.approx((1.0 * 0.25 + 0.25 * 3.0 + 0.75 * 2.0 + 0.5 * 2.0) / 7.25)


def test_league_eval_warmup_gate_requires_frontier_anchor_thresholds() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                enabled=True,
                warmup=SimpleNamespace(
                    eval_gate_enabled=True,
                    eval_gate_min_aggregate_score=0.57,
                    eval_gate_min_anchor_scores={
                        "B1 NoLeague baseline": 0.45,
                        "B3 HeuristicPublicAggro": 0.60,
                    },
                ),
            )
        )
    )

    closed = _league_eval_warmup_gate_status(
        stack,
        {
            "aggregate_score": 0.58,
            "anchor_scores": {
                "B1 NoLeague baseline": 0.4375,
                "B3 HeuristicPublicAggro": 0.625,
            },
        },
    )
    opened = _league_eval_warmup_gate_status(
        stack,
        {
            "aggregate_score": 0.58,
            "anchor_scores": {
                "B1 NoLeague baseline": 0.5,
                "B3 HeuristicPublicAggro": 0.625,
            },
        },
    )

    assert closed["enabled"] is True
    assert closed["open"] is False
    assert closed["failed_anchors"]["B1 NoLeague baseline"]["score"] == pytest.approx(0.4375)
    assert opened["open"] is True


def test_league_eval_warmup_gate_ignores_unavailable_symbolic_recent_anchor() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                enabled=True,
                warmup=SimpleNamespace(
                    eval_gate_enabled=True,
                    eval_gate_min_aggregate_score=0.57,
                    eval_gate_min_anchor_scores={
                        "B1 NoLeague baseline": 0.50,
                        "B3 HeuristicPublicAggro": 0.90,
                        "Previous recent snapshot": 0.50,
                    },
                ),
            )
        )
    )

    status = _league_eval_warmup_gate_status(
        stack,
        {
            "aggregate_score": 0.72,
            "anchor_scores": {
                "B1 NoLeague baseline": 0.5625,
                "B3 HeuristicPublicAggro": 0.9375,
            },
        },
    )

    assert status["open"] is True
    assert status["failed_anchors"] == {}


def test_checkpoint_candidate_metric_rejects_truncation_heavy_dev_eval() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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


def test_periodic_dev_eval_previous_recent_ignores_seed_imports_and_rejections(tmp_path: Path) -> None:
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
    registry = SnapshotRegistry(recent_size=8, champion_size=1)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
        source_kind="baseline_anchor",
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
    registry.add_snapshot(
        policy_id="seed_imported_policy_000009",
        update=450,
        weights_sha256="e" * 64,
        path=snapshot_weights_relpath("seed_imported_policy_000009"),
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="policy_000040",
        update=40,
        weights_sha256="f" * 64,
        path=snapshot_weights_relpath("policy_000040"),
    )
    registry.reject_snapshot("policy_000040")
    registry.save(snapshots_dir / "registry.json")

    specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(stack=stack, run_dir=tmp_path)

    assert specs[-1] == PeriodicDevEvalOpponentSpec(
        policy_id="policy_000020",
        display_name="Previous recent snapshot",
        kind="snapshot",
        snapshot_path=snapshot_weights_relpath("policy_000020"),
        heuristic_profile=None,
    )
    assert pinned_snapshot_ids[-1] == "policy_000020"


def test_periodic_dev_eval_previous_recent_requires_champion_when_promotion_gated(tmp_path: Path) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_gate_enabled=True,
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=("Previous recent snapshot",),
                ),
            )
        )
    )
    snapshots_dir = tmp_path / "training" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    registry = SnapshotRegistry(recent_size=8, champion_size=2)
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=5,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
        source_kind="baseline_anchor",
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
    registry.add_champion("policy_000030")
    registry.save(snapshots_dir / "registry.json")

    specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(stack=stack, run_dir=tmp_path)

    assert specs[-1] == PeriodicDevEvalOpponentSpec(
        policy_id="policy_000010",
        display_name="Previous recent snapshot",
        kind="snapshot",
        snapshot_path=snapshot_weights_relpath("policy_000010"),
        heuristic_profile=None,
    )
    assert pinned_snapshot_ids[-1] == "policy_000010"


def test_symbolic_snapshot_aliases_keep_seed_history_explicit() -> None:
    registry = SnapshotRegistry(recent_size=8, champion_size=2)
    registry.add_snapshot(
        policy_id="seed_imported_policy_000009",
        update=450,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("seed_imported_policy_000009"),
        source_kind="seed_import",
    )
    registry.add_snapshot(
        policy_id="policy_000480",
        update=480,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("policy_000480"),
        source_kind="league_import",
    )
    registry.add_snapshot(
        policy_id="policy_000500",
        update=500,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000500"),
    )
    registry.add_champion("seed_imported_policy_000009")
    registry.add_champion("policy_000480")

    assert train_script._resolve_symbolic_promotion_anchor_policy_id(
        "Latest imported seed history snapshot",
        registry=registry,
    ) == "seed_imported_policy_000009"
    assert train_script._resolve_symbolic_promotion_anchor_policy_id(
        "Latest local candidate snapshot",
        registry=registry,
    ) == "policy_000500"
    assert train_script._resolve_symbolic_promotion_anchor_policy_id(
        "Latest promoted champion snapshot",
        registry=registry,
    ) == "policy_000480"
    assert train_script._resolve_symbolic_promotion_anchor_policy_id(
        "Latest champion snapshot",
        registry=registry,
    ) == "policy_000480"


def test_pin_and_unpin_snapshot_ids_preserve_existing_pins(tmp_path: Path) -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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


def test_resolved_periodic_dev_eval_worker_devices_cycles_explicit_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = SimpleNamespace(config=SimpleNamespace(system=None))
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_script.torch.cuda, "device_count", lambda: 3)

    resolved = _resolved_periodic_dev_eval_worker_devices(
        stack=stack,
        parallel_workers=4,
        explicit_worker_devices=("cuda:0", "cuda:2"),
        eval_device="cuda:auto",
        learner_device=None,
    )

    assert resolved == ("cuda:0", "cuda:2", "cuda:0", "cuda:2")


def test_dev_eval_ineligibility_reasons_identify_borderline_confidence_only() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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


def test_dev_eval_ineligibility_allows_exact_tied_anchor_without_loss_probability() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.8333333333333334,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.0}},
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.0, "prob_lt_half": 0.0, "ci_half_width": 0.0}
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert reasons == ()


def test_dev_eval_ineligibility_rejects_anchor_with_loss_probability() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

    reasons = _dev_eval_ineligibility_reasons(
        stack,
        dev_eval_summary={
            "aggregate_score": 0.7589285714285714,
            "anchors": {
                "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.0}},
                "Previous champion snapshot": {
                    "uncertainty": {
                        "prob_gt_half": 0.0,
                        "prob_lt_half": 1.0,
                        "ci_half_width": 0.10035994480847998,
                    }
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert reasons == ("confidence_prob",)


def test_confirmatory_dev_eval_request_targets_borderline_score_drop_for_reevaluation() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

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


def test_fast_batched_dev_eval_screen_is_non_authoritative() -> None:
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    summary = {
        "aggregate_score": 0.75,
        "evaluation_surface": {
            "kind": "fast_batched_screen",
            "authoritative": False,
            "batched_inference_enabled": True,
        },
        "anchors": {
            "B0 RandomLegal": {"uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.05}},
        },
    }

    assert train_script._dev_eval_is_authoritative(summary) is False
    assert _dev_eval_ineligibility_reasons(stack, dev_eval_summary=summary) == ("non_authoritative",)


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
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
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


def test_reference_policy_bc_coefs_for_next_update_linearly_anneal() -> None:
    training = SimpleNamespace(
        reference_policy_top_action_bc_coef=0.5,
        reference_policy_top_action_bc_final_coef=0.05,
        reference_policy_top_action_bc_start_updates=800,
        reference_policy_top_action_bc_end_updates=900,
        reference_policy_top_action_family_bc_coef=0.75,
        reference_policy_top_action_family_bc_final_coef=0.0,
        reference_policy_top_action_family_bc_start_updates=800,
        reference_policy_top_action_family_bc_end_updates=900,
    )

    assert _reference_policy_top_action_bc_coef_for_next_update(training, update_count=799) == pytest.approx(0.5)
    assert _reference_policy_top_action_bc_coef_for_next_update(training, update_count=850) == pytest.approx(0.275)
    assert _reference_policy_top_action_bc_coef_for_next_update(training, update_count=901) == pytest.approx(0.05)
    assert _reference_policy_top_action_family_bc_coef_for_next_update(training, update_count=850) == pytest.approx(
        0.375
    )


def test_persist_periodic_dev_eval_summary_round_trips_anchor_payloads_and_b2_warnings(tmp_path: Path) -> None:
    training_paths = _training_paths(tmp_path)

    for update_count, policy_version, b2_score in ((10, 0, 0.01), (20, 1, 0.0), (30, 2, 0.0)):
        train_script._persist_periodic_dev_eval_summary(
            training_paths=training_paths,
            payload={
                "policy_id": f"train_u{update_count}_p{policy_version}",
                "update_count": update_count,
                "policy_version": policy_version,
                "aggregate_score": 0.3,
                "unweighted_aggregate_score": 0.4,
                "aggregate_weighting": {
                    "version": "periodic_dev_eval_anchor_weights_v1",
                    "anchor_weights": {"B2 HeuristicPublic": 3.0},
                    "total_weight": 3.0,
                    "default_weight": 1.0,
                },
                "anchor_scores": {
                    "B0 RandomLegal": 1.0,
                    "B1 NoLeague baseline": 0.5,
                    "B2 HeuristicPublic": b2_score,
                },
                "anchors": {
                    "B2 HeuristicPublic": {
                        "summary": {
                            "games": 16,
                            "total_actions": 100,
                            "main_move_actions": 60,
                            "pass_with_nonpass_available": 10,
                            "max_consecutive_main_moves": 8,
                        },
                        "uncertainty": {
                            "mean": b2_score,
                            "ci_half_width": 0.05,
                            "prob_gt_half": 0.05,
                            "prob_lt_half": 0.95,
                        },
                        "evaluation_context": {
                            "episodes_path": f"eval/dev_eval/update_{update_count}/B2 HeuristicPublic/episodes.jsonl"
                        },
                    }
                },
                "stall_monitor": {"worst_anchor": "B2 HeuristicPublic", "worst_stall_rate": 0.0},
            },
        )

    persisted = json.loads(train_script._periodic_dev_eval_summaries_path(training_paths).read_text(encoding="utf-8"))
    latest = persisted["train_u30_p2"]
    assert latest["format"] == "periodic_dev_eval_summary_v2"
    assert latest["unweighted_aggregate_score"] == pytest.approx(0.4)
    assert latest["aggregate_weighting"]["anchor_weights"]["B2 HeuristicPublic"] == pytest.approx(3.0)
    assert latest["stall_monitor"]["worst_anchor"] == "B2 HeuristicPublic"
    assert latest["b2"]["score"] == pytest.approx(0.0)
    warning_kinds = {entry["kind"] for entry in latest["b2"]["warning_flags"]}
    assert "b2_flatline_v1" in warning_kinds
    assert "b2_action_family_warning_v1" in warning_kinds
    assert latest["anchors"]["B2 HeuristicPublic"]["evaluation_context"]["episodes_path"].endswith(
        "update_30/B2 HeuristicPublic/episodes.jsonl"
    )


def test_publish_best_checkpoint_from_dev_eval_tracks_secondary_best_b2(tmp_path: Path) -> None:
    training_paths = _training_paths(tmp_path)
    artifacts = SimpleNamespace(run_dir=tmp_path)
    checkpoint_a = training_paths.checkpoints_dir / "checkpoint_a.pt"
    checkpoint_b = training_paths.checkpoints_dir / "checkpoint_b.pt"
    checkpoint_a.write_text("a", encoding="utf-8")
    checkpoint_b.write_text("b", encoding="utf-8")
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")

    train_script._publish_best_checkpoint_from_dev_eval(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_a,
        update_count=10,
        policy_version=1,
        dev_eval_summary={
            "aggregate_score": 0.55,
            "anchor_scores": {"B2 HeuristicPublic": 0.25},
            "anchors": {"B2 HeuristicPublic": {"uncertainty": {"mean": 0.25}}},
        },
    )
    train_script._publish_best_checkpoint_from_dev_eval(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_b,
        update_count=20,
        policy_version=2,
        dev_eval_summary={
            "aggregate_score": 0.52,
            "anchor_scores": {"B2 HeuristicPublic": 0.5},
            "anchors": {"B2 HeuristicPublic": {"uncertainty": {"mean": 0.5}}},
        },
    )

    tracker = train_script._load_checkpoint_tracker(training_paths)
    assert tracker["format"] == "checkpoint_tracker_v2"
    assert tracker["secondary"]["best_b2"]["metric_kind"] == "b2_score"
    assert tracker["secondary"]["best_b2"]["metric_value"] == pytest.approx(0.5)
    assert tracker["secondary"]["best_b2"]["source_checkpoint_path"] == "training/checkpoints/checkpoint_b.pt"


def test_specialized_noleague_baseline_roles_are_accepted_for_train_and_eval_imports() -> None:
    config_canonical = {
        "config": {
            "experiment": {
                "role": "baseline_noleague_ablation_teacher_fade_no_tactical_bias",
            }
        }
    }

    train_script._assert_noleague_baseline_config(config_canonical)
    assert simulator_runner._config_marks_noleague_baseline(config_canonical) is True


def test_wall_clock_budget_reached_only_after_elapsed_budget() -> None:
    start_time = 100.0
    assert train_script._wall_clock_budget_seconds(7.5) == pytest.approx(450.0)
    assert train_script._wall_clock_budget_reached(start_time=start_time, max_wall_clock_seconds=450.0, now=549.9) is False
    assert train_script._wall_clock_budget_reached(start_time=start_time, max_wall_clock_seconds=450.0, now=550.0) is True
    assert train_script._wall_clock_budget_reached(start_time=start_time, max_wall_clock_seconds=None, now=999.0) is False


def test_maybe_request_b2_disagreement_audit_on_flatline_writes_request(tmp_path: Path) -> None:
    training_paths = _training_paths(tmp_path)
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    episodes_path = tmp_path / "eval" / "dev_eval" / "update_30" / "B2 HeuristicPublic" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("{}\n", encoding="utf-8")
    stack_config_path = _repo_root() / "configs" / "local.yaml"
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(json.dumps({"stack_config_path": stack_config_path.as_posix()}), encoding="utf-8")
    artifacts = SimpleNamespace(run_dir=tmp_path, run_summary_path=run_summary_path)

    payload = train_script._maybe_request_b2_disagreement_audit(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        dev_eval_summary={
            "policy_id": "train_u30_p2",
            "update_count": 30,
            "policy_version": 2,
            "aggregate_score": 0.52,
            "anchor_scores": {"B2 HeuristicPublic": 0.2},
            "anchors": {
                "B2 HeuristicPublic": {
                    "uncertainty": {"mean": 0.2, "ci_half_width": 0.02, "prob_gt_half": 0.9},
                    "evaluation_context": {
                        "episodes_path": "eval/dev_eval/update_30/B2 HeuristicPublic/episodes.jsonl",
                    },
                }
            },
            "b2": {"warning_flags": [{"kind": "b2_flatline_v1"}]},
        },
    )

    assert payload is not None
    assert payload["trigger_reasons"] == ["b2_flatline"]
    assert payload["episodes_path"] == "eval/dev_eval/update_30/B2 HeuristicPublic/episodes.jsonl"
    assert payload["command"][1] == "python/scripts/b2_disagreement_audit.py"
    request_log = train_script._b2_disagreement_audit_requests_path(training_paths)
    assert request_log.is_file()
    entries = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries[-1]["trigger_reasons"] == ["b2_flatline"]


def test_maybe_request_b2_disagreement_audit_on_confidence_only_gate(tmp_path: Path) -> None:
    training_paths = _training_paths(tmp_path)
    stack = load_stack_config(_repo_root() / "configs" / "local.yaml")
    episodes_path = tmp_path / "eval" / "dev_eval" / "update_40" / "B2 HeuristicPublic" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("{}\n", encoding="utf-8")
    stack_config_path = _repo_root() / "configs" / "local.yaml"
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(json.dumps({"stack_config_path": stack_config_path.as_posix()}), encoding="utf-8")
    artifacts = SimpleNamespace(run_dir=tmp_path, run_summary_path=run_summary_path)

    payload = train_script._maybe_request_b2_disagreement_audit(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        dev_eval_summary={
            "policy_id": "train_u40_p3",
            "update_count": 40,
            "policy_version": 3,
            "aggregate_score": 0.58,
            "anchor_scores": {
                "B0 RandomLegal": 1.0,
                "B1 NoLeague baseline": 0.6,
                "B2 HeuristicPublic": 0.32,
            },
            "anchors": {
                "B0 RandomLegal": {
                    "uncertainty": {"mean": 1.0, "ci_half_width": 0.01, "prob_gt_half": 1.0},
                },
                "B1 NoLeague baseline": {
                    "uncertainty": {"mean": 0.6, "ci_half_width": 0.02, "prob_gt_half": 0.95},
                },
                "B2 HeuristicPublic": {
                    "uncertainty": {"mean": 0.32, "ci_half_width": 0.25, "prob_gt_half": 0.2},
                    "evaluation_context": {
                        "episodes_path": "eval/dev_eval/update_40/B2 HeuristicPublic/episodes.jsonl",
                    },
                },
            },
            "b2": {"warning_flags": []},
        },
    )

    assert payload is not None
    assert payload["trigger_reasons"] == ["confidence_only_gate"]
    request_log = train_script._b2_disagreement_audit_requests_path(training_paths)
    entries = [json.loads(line) for line in request_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries[-1]["trigger_reasons"] == ["confidence_only_gate"]

