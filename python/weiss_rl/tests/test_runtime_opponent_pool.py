from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig


def test_refresh_opponent_pool_excludes_fixed_b1_anchor(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._pfsp_pool_size == 1
    assert runtime_any._opponent_models == {"policy_000007": "loaded::training/snapshots/policy_000007/weights.pt"}


def test_refresh_opponent_pool_keeps_reserved_b1_anchor_resident(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 1
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "b1_noleague_baseline": "loaded::training/snapshots/b1_noleague_baseline/weights.pt",
    }


def test_refresh_opponent_pool_keeps_mixed_b1_anchor_resident(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
        sampling=SimpleNamespace(
            noleague_baseline_mix_fraction=0.15,
            noleague_baseline_mix_end_updates=-1,
        ),
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "b1_noleague_baseline": "loaded::training/snapshots/b1_noleague_baseline/weights.pt",
    }


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000007",)
    assert runtime_any._opponent_recent_ids == ("policy_000008",)
    assert runtime_any._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime_any._pfsp_pool_size == 2
    assert runtime_any._pfsp_recent_pool_size == 1
    assert runtime_any._opponent_models == {
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
    }


def test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ()
    assert runtime_any._opponent_recent_ids == ("policy_000008",)
    assert runtime_any._opponent_candidate_ids == ("policy_000008",)
    assert runtime_any._pfsp_pool_size == 1
    assert runtime_any._pfsp_recent_pool_size == 1
    assert runtime_any._opponent_models == {"policy_000008": "loaded::training/snapshots/policy_000008/weights.pt"}


def test_refresh_opponent_pool_keeps_models_for_inflight_stale_assignments(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=999,
        weights_sha256="b" * 64,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._actors = [SimpleNamespace(opponent_policy_id_by_env=np.asarray(["policy_000007"], dtype=object))]
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000008",)
    assert runtime_any._opponent_models == {
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
    }


def test_maybe_publish_snapshot_tracks_effective_update_for_reused_weights() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")
    runtime_any._collector_result_queue = None
    runtime_any._collector_control_queues = []
    runtime_any._collector_free_queues = []
    runtime_any._shared_actor_model = None
    runtime_any._bootstrap_models = None
    runtime_any._actors = [SimpleNamespace(model=torch.nn.Linear(2, 2), snapshot_version=0)]
    runtime_any._league_enabled = True
    runtime_any._league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=200),
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        promotion_gate_enabled=True,
    )
    runtime_any._opponent_sampler = object()
    runtime_any._opponent_candidate_ids = ("policy_000007",)
    runtime_any._opponent_models = {"policy_000007": object()}
    runtime_any._current_learner_update = 0
    runtime_any._effective_learner_update = 0
    runtime_any._published_snapshot_update_by_fingerprint = {}
    runtime_any._last_published_snapshot_version = 0

    learner_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        learner_model.weight.fill_(1.0)
        learner_model.bias.fill_(0.5)

    QueueRuntime.maybe_publish_snapshot(
        runtime, learner_model=cast(Any, learner_model), learner_update_count=20, force=True
    )
    assert runtime_any._current_learner_update == 20
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False

    restored_model = torch.nn.Linear(2, 2)
    restored_model.load_state_dict(learner_model.state_dict())

    QueueRuntime.maybe_publish_snapshot(
        runtime,
        learner_model=cast(Any, restored_model),
        learner_update_count=220,
        force=True,
    )
    assert runtime_any._current_learner_update == 220
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False


def test_refresh_opponent_pool_uses_effective_update_for_champion_age(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000120",
        update=120,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000120/weights.pt",
    )
    registry.add_champion("policy_000120")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        pool=SimpleNamespace(champion_max_age_updates=50),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._current_learner_update = 220
    runtime_any._effective_learner_update = 20
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000120",)
    assert runtime_any._opponent_candidate_ids == ("policy_000120",)


def test_refresh_opponent_pool_quarantines_timeout_heavy_champions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.add_champion("policy_000008")
    registry.save(registry_path)

    outcomes = OnlineOutcomeTracker(window_size=128)
    for _ in range(40):
        outcomes.update("policy_000007", "t")
        outcomes.update("policy_000008", "w")

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = outcomes
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime_any._pfsp_pool_size == 2
    assert runtime_any._pfsp_quarantined_opponents == 1
    assert runtime_any._opponent_models == {
        "policy_000008": "loaded::training/snapshots/policy_000008/weights.pt",
        "policy_000007": "loaded::training/snapshots/policy_000007/weights.pt",
    }


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_champions_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000190",
        update=190,
        weights_sha256="a" * 64,
        path="training/snapshots/policy_000190/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000191",
        update=191,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000191/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000192",
        update=192,
        weights_sha256="c" * 64,
        path="training/snapshots/policy_000192/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000193",
        update=193,
        weights_sha256="d" * 64,
        path="training/snapshots/policy_000193/weights.pt",
    )
    registry.add_champion("policy_000190")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_champion_ids == ("policy_000190",)
    assert runtime_any._opponent_recent_ids == ("policy_000192", "policy_000193")
    assert runtime_any._opponent_candidate_ids == ("policy_000190", "policy_000192", "policy_000193")
    assert runtime_any._pfsp_pool_size == 3
    assert runtime_any._pfsp_recent_pool_size == 2
    assert runtime_any._opponent_models == {
        "policy_000190": "loaded::training/snapshots/policy_000190/weights.pt",
        "policy_000192": "loaded::training/snapshots/policy_000192/weights.pt",
        "policy_000193": "loaded::training/snapshots/policy_000193/weights.pt",
    }


def test_refresh_opponent_pool_demotes_stale_champions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000010",
        update=10,
        weights_sha256="a" * 64,
        path="training/snapshots/policy_000010/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000190",
        update=190,
        weights_sha256="b" * 64,
        path="training/snapshots/policy_000190/weights.pt",
    )
    registry.add_champion("policy_000010")
    registry.add_champion("policy_000190")
    registry.save(registry_path)

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._current_learner_update = 220
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=4,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=True,
        pool=SimpleNamespace(champion_max_age_updates=40),
        sampling=SimpleNamespace(
            champion_mix_fraction=0.35,
            hard_negative_mix_fraction=0.2,
            hard_negative_min_samples=16,
            hard_negative_max_win_rate=0.45,
        ),
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    refreshed = SnapshotRegistry.load(registry_path)
    assert refreshed.champion_snapshots == ["policy_000190"]
    assert runtime_any._opponent_champion_ids == ("policy_000190",)
    assert runtime_any._opponent_recent_ids == ("policy_000010",)
    assert runtime_any._opponent_candidate_ids == ("policy_000190", "policy_000010")


def test_refresh_opponent_pool_writes_pool_composition_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000007",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/policy_000007/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000008",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000008/weights.pt",
    )
    registry.add_champion("policy_000007")
    registry.add_champion("policy_000008")
    registry.save(registry_path)

    outcomes = OnlineOutcomeTracker(window_size=128)
    outcomes.update("policy_000007", "l")
    outcomes.update("policy_000007", "l")

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._run_dir = run_dir
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._current_learner_update = 11
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
        sampling=SimpleNamespace(
            hard_negative_min_samples=2,
            hard_negative_max_win_rate=0.5,
            hard_negative_focus_policy_ids=("policy_000007",),
            hard_negative_focus_weight_multiplier=2.0,
            row_deficit_policy_weights=(("policy_000008", 3.0),),
        ),
    )
    runtime_any._outcomes = outcomes
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

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
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="champion_weak",
        update=7,
        weights_sha256="7" * 64,
        path="training/snapshots/champion_weak/weights.pt",
    )
    registry.add_snapshot(
        policy_id="champion_solid",
        update=8,
        weights_sha256="8" * 64,
        path="training/snapshots/champion_solid/weights.pt",
    )
    registry.add_champion("champion_weak")
    registry.add_champion("champion_solid")
    registry.save(registry_path)

    outcomes = OnlineOutcomeTracker(window_size=128)
    outcomes.update("champion_weak", "l")
    outcomes.update("champion_weak", "l")
    outcomes.update("champion_solid", "w")
    outcomes.update("champion_solid", "w")

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._run_dir = run_dir
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._current_learner_update = 11
    runtime_any._league_config = SimpleNamespace(
        snapshot_pool_recent_size=8,
        snapshot_pool_champion_size=2,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=False,
        sampling=SimpleNamespace(
            hard_negative_min_samples=2,
            hard_negative_max_win_rate=0.5,
            hard_negative_focus_policy_ids=(),
            hard_negative_focus_weight_multiplier=1.0,
            hard_negative_overlaps_champions=True,
        ),
    )
    runtime_any._outcomes = outcomes
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = 0
    runtime_any._noleague_baseline_reserved_envs_per_actor = 0
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_hard_negative_ids == ("champion_weak",)
    assert runtime_any._opponent_champion_ids == ("champion_weak", "champion_solid")
    assert runtime_any._opponent_candidate_ids == ("champion_weak", "champion_solid")
    assert runtime_any._pfsp_hard_negative_pool_size == 1
    assert runtime_any._pfsp_champion_pool_size == 2

    log_path = run_dir / "training" / "logs" / "opponent_pool.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["hard_negative_overlaps_champions"] is True
    assert record["hard_negative_ids"] == ["champion_weak"]
    assert record["champion_ids"] == ["champion_weak", "champion_solid"]
