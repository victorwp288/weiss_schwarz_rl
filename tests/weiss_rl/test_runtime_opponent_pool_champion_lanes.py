from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.runtime import QueueRuntime

from .runtime_opponent_pool_test_support import (
    loaded_snapshot_models,
    make_opponent_pool_runtime,
    opponent_pool_config,
    outcomes_from_results,
    write_snapshot_registry,
)


def test_refresh_opponent_pool_quarantines_timeout_heavy_champions(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [("policy_000007", 7), ("policy_000008", 8)],
        champions=("policy_000007", "policy_000008"),
    )
    outcomes = OnlineOutcomeTracker(window_size=128)
    for _ in range(40):
        outcomes.update("policy_000007", "t")
        outcomes.update("policy_000008", "w")
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(promotion_gate_enabled=True),
        outcomes=outcomes,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime._pfsp_pool_size == 2
    assert runtime._pfsp_quarantined_opponents == 1
    assert runtime._opponent_models == loaded_snapshot_models("policy_000008", "policy_000007")


def test_refresh_opponent_pool_demotes_stale_champions(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [("policy_000010", 10), ("policy_000190", 190)],
        champions=("policy_000010", "policy_000190"),
    )
    sampling = SimpleNamespace(
        champion_mix_fraction=0.35,
        hard_negative_mix_fraction=0.2,
        hard_negative_min_samples=16,
        hard_negative_max_win_rate=0.45,
    )
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(
            champion_size=4,
            promotion_gate_enabled=True,
            sampling=sampling,
            pool=SimpleNamespace(champion_max_age_updates=40),
        ),
        current_update=220,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    refreshed = SnapshotRegistry.load(registry_path)
    assert refreshed.champion_snapshots == ["policy_000190"]
    assert runtime._opponent_champion_ids == ("policy_000190",)
    assert runtime._opponent_recent_ids == ("policy_000010",)
    assert runtime._opponent_candidate_ids == ("policy_000190", "policy_000010")


def test_refresh_opponent_pool_writes_pool_composition_log(tmp_path: Path) -> None:
    run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [("policy_000007", 7), ("policy_000008", 8)],
        champions=("policy_000007", "policy_000008"),
    )
    outcomes = outcomes_from_results("policy_000007", "l", 2)
    sampling = SimpleNamespace(
        hard_negative_min_samples=2,
        hard_negative_max_win_rate=0.5,
        hard_negative_focus_policy_ids=("policy_000007",),
        hard_negative_focus_weight_multiplier=2.0,
        row_deficit_policy_weights=(("policy_000008", 3.0),),
    )
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(sampling=sampling),
        run_dir=run_dir,
        outcomes=outcomes,
        current_update=11,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    log_path = run_dir / "training" / "logs" / "opponent_pool.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "opponent_pool_refresh_v1"
    assert record["reason"] == "refreshed"
    assert record["update"] == 11
    assert record["registry_path"] == "training/snapshots/registry.json"
    assert record["hard_negative_ids"] == ["policy_000007", "policy_000008"]
    assert record["hard_negative_focus_policy_ids"] == ["policy_000007"]
    assert record["hard_negative_focus_weight_multiplier"] == 2.0
    assert record["row_deficit_policy_weights"] == [["policy_000008", 3.0]]
    assert record["champion_ids"] == []
    assert record["candidate_ids"] == ["policy_000007", "policy_000008"]
    assert record["champion_pool_size"] == 0
    assert record["hard_negative_pool_size"] == 2
    assert record["loaded_model_ids"] == ["policy_000007", "policy_000008"]


def test_refresh_opponent_pool_can_keep_hard_negative_champions_in_both_lanes(tmp_path: Path) -> None:
    run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [("champion_weak", 7), ("champion_solid", 8)],
        champions=("champion_weak", "champion_solid"),
    )
    outcomes = OnlineOutcomeTracker(window_size=128)
    outcomes.update("champion_weak", "l")
    outcomes.update("champion_weak", "l")
    outcomes.update("champion_solid", "w")
    outcomes.update("champion_solid", "w")
    sampling = SimpleNamespace(
        hard_negative_min_samples=2,
        hard_negative_max_win_rate=0.5,
        hard_negative_focus_policy_ids=(),
        hard_negative_focus_weight_multiplier=1.0,
        hard_negative_overlaps_champions=True,
    )
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(sampling=sampling),
        run_dir=run_dir,
        outcomes=outcomes,
        current_update=11,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_hard_negative_ids == ("champion_weak",)
    assert runtime._opponent_champion_ids == ("champion_weak", "champion_solid")
    assert runtime._opponent_candidate_ids == ("champion_weak", "champion_solid")
    assert runtime._pfsp_hard_negative_pool_size == 1
    assert runtime._pfsp_champion_pool_size == 2

    log_path = run_dir / "training" / "logs" / "opponent_pool.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["hard_negative_overlaps_champions"] is True
    assert record["hard_negative_ids"] == ["champion_weak"]
    assert record["champion_ids"] == ["champion_weak", "champion_solid"]
