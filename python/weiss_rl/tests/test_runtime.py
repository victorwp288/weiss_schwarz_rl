from __future__ import annotations

import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.runtime import (
    QueueRuntime,
    QueueRuntimeConfig,
    RuntimeUnroll,
    _MIRROR_OPPONENT_POLICY_ID,
    _create_shared_collector_slot_config,
    _concatenate_legal_actions,
    _gae_advantages,
    _open_shared_collector_slot,
    _read_unroll_from_shared_slot,
    _resolve_actor_topology,
    _shared_unroll_metadata,
    _write_unroll_to_shared_slot,
    build_runtime_config,
)


def _make_runtime_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    counters: dict[str, int] | None = None,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=f"{actor_id}:{unroll_seq}:{behavior_policy_version}",
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=np.zeros((1, 1), dtype=np.bool_),
        to_play_seat=np.zeros((1, 1), dtype=np.int64),
        behavior_logp=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
        behavior_logits=None,
        counters=counters,
    )


def test_select_pending_unrolls_train_ordered_keeps_same_behavior_version() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=2,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=3,
        queue_capacity_unrolls=3,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._pending_unrolls = deque(
        [
            _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
            _make_runtime_unroll(actor_id=0, unroll_seq=1, behavior_policy_version=1),
            _make_runtime_unroll(actor_id=1, unroll_seq=1, behavior_policy_version=1),
        ]
    )

    selected = QueueRuntime._select_pending_unrolls(runtime)

    assert [(item.behavior_policy_version, item.unroll_seq, item.actor_id) for item in selected] == [
        (0, 0, 0),
        (0, 0, 1),
    ]


def test_reset_done_rows_fallback_reinitializes_full_actor_state() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=2,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._device = torch.device("cpu")

    assign_calls: list[tuple[np.ndarray, bool]] = []

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        assign_calls.append((np.asarray(done, dtype=np.bool_).copy(), initial))
        actor.focal_seat_by_env[:] = np.array([0, 1], dtype=np.int64)
        actor.opponent_policy_id_by_env[:] = np.asarray(["mirror", "policy_000007"], dtype=object)

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done) -> None:
            raise RuntimeError("reset_done unsupported")

        def reset(self, *, seed: int):
            return {"seed": seed}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((2, 3)),
            opponent_hidden=torch.full((2, 3), 2.0),
            focal_seat_by_env=np.array([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["stale_a", "stale_b"], dtype=object),
        ),
    )

    batch = cast(Any, QueueRuntime._reset_done_rows(runtime, actor, np.array([True, False], dtype=np.bool_)))

    assert isinstance(batch, dict)
    assert isinstance(batch["seed"], int)
    assert torch.count_nonzero(actor.seat_hidden) == 0
    assert torch.count_nonzero(actor.opponent_hidden) == 0
    assert len(assign_calls) == 1
    assert np.array_equal(assign_calls[0][0], np.array([True, True], dtype=np.bool_))
    assert assign_calls[0][1] is True
    assert actor.focal_seat_by_env.tolist() == [0, 1]
    assert actor.opponent_policy_id_by_env.tolist() == ["mirror", "policy_000007"]


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

    QueueRuntime.maybe_publish_snapshot(runtime, learner_model=learner_model, learner_update_count=20, force=True)
    assert runtime_any._current_learner_update == 20
    assert runtime_any._effective_learner_update == 20
    assert QueueRuntime._pfsp_sampling_ready(runtime) is False

    restored_model = torch.nn.Linear(2, 2)
    restored_model.load_state_dict(learner_model.state_dict())

    QueueRuntime.maybe_publish_snapshot(runtime, learner_model=restored_model, learner_update_count=220, force=True)
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


def test_sample_opponent_policy_ids_can_force_hard_negative_bucket() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ("policy_hard", "policy_recent")
    runtime_any._opponent_hard_negative_ids = ("policy_hard",)
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ("policy_recent",)
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=0.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=1.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {"policy_hard": object(), "policy_recent": object()}
    runtime_any._pfsp_sampling_ready = lambda: True

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == ("policy_hard", "policy_hard", "policy_hard", "policy_hard")
    assert runtime_any._pfsp_last_hard_negative_envs == 4
    assert runtime_any._pfsp_last_recent_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 0


def test_sample_opponent_policy_ids_can_force_heuristic_public_bucket_before_pfsp_ready() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._league_enabled = True
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_hard_negative_ids = ()
    runtime_any._opponent_champion_ids = ()
    runtime_any._opponent_recent_ids = ()
    runtime_any._opponent_heuristic_policies = {"B2 HeuristicPublic": object()}
    runtime_any._league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            heuristic_public_mix_fraction=1.0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )
    runtime_any._outcomes = OnlineOutcomeTracker(window_size=128)
    runtime_any._opponent_models = {}
    runtime_any._pfsp_sampling_ready = lambda: False
    runtime_any._league_reference_update = lambda: 0

    sampled = QueueRuntime._sample_opponent_policy_ids(
        runtime,
        count=4,
        rng=np.random.default_rng(7),
    )

    assert sampled == (
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
        "B2 HeuristicPublic",
    )
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_mirror_envs == 0
    assert runtime_any._pfsp_last_heuristic_public_envs == 4


def test_assign_episode_roles_prioritizes_fixed_anchor_lanes() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._pfsp_last_sampled_envs = 0
    runtime_any._pfsp_last_mirror_envs = 0
    runtime_any._pfsp_last_heuristic_public_envs = 0
    runtime_any._pfsp_last_noleague_baseline_envs = 0
    runtime_any._pfsp_last_champion_envs = 0
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 0
    runtime_any._fixed_opponent_policy_is_active = lambda policy_id: True

    def fake_sample(*, count: int, rng) -> tuple[str, ...]:
        runtime_any._pfsp_last_sampled_envs = count
        runtime_any._pfsp_last_mirror_envs = 0
        runtime_any._pfsp_last_heuristic_public_envs = 0
        runtime_any._pfsp_last_noleague_baseline_envs = 0
        runtime_any._pfsp_last_champion_envs = 0
        runtime_any._pfsp_last_recent_envs = count
        runtime_any._pfsp_last_hard_negative_envs = 0
        return tuple(f"recent_{index}" for index in range(count))

    runtime_any._sample_opponent_policy_ids = fake_sample

    actor = cast(
        Any,
        SimpleNamespace(
            actor_id=0,
            rng=np.random.default_rng(7),
            focal_seat_by_env=np.asarray([0, 1, 0, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["old0", "old1", "old2", "old3"], dtype=object),
            fixed_opponent_policy_id_by_env=np.asarray(
                ["B2 HeuristicPublic", "b1_noleague_baseline", "", ""],
                dtype=object,
            ),
        ),
    )

    QueueRuntime._assign_episode_roles(runtime, actor, np.asarray([True, True, True, True], dtype=np.bool_), initial=False)

    assert actor.opponent_policy_id_by_env.tolist() == [
        "B2 HeuristicPublic",
        "b1_noleague_baseline",
        "recent_0",
        "recent_1",
    ]
    assert runtime_any._pfsp_last_sampled_envs == 4
    assert runtime_any._pfsp_last_heuristic_public_envs == 1
    assert runtime_any._pfsp_last_noleague_baseline_envs == 1
    assert runtime_any._pfsp_last_recent_envs == 2


def test_overwrite_central_outputs_with_opponents_only_touches_non_mirror_rows() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}
    runtime_any._opponent_heuristic_policies = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.full((obs_tensor.shape[0], 5), -7.0, dtype=torch.float32)
            values = torch.full((obs_tensor.shape[0],), 42.0, dtype=torch.float32)
            return logits, values, hidden_tensor + 1.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor = cast(
        Any,
        SimpleNamespace(
            layout_name="i16_legal_ids",
            focal_seat_by_env=np.asarray([0, 0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(
                [_MIRROR_OPPONENT_POLICY_ID, "policy_000007", _MIRROR_OPPONENT_POLICY_ID],
                dtype=object,
            ),
            opponent_hidden=torch.zeros((3, 4)),
            rng=np.random.default_rng(7),
        ),
    )
    batch = cast(
        Any,
        SimpleNamespace(
            obs=np.zeros((3, 8), dtype=np.float32),
            actor=np.asarray([1, 1, 1], dtype=np.int64),
            ids_offsets=(
                np.asarray([0, 1, 2], dtype=np.uint32),
                np.asarray([0, 1, 2, 3], dtype=np.uint32),
            ),
            mask=None,
        ),
    )
    logits = np.zeros((3, 5), dtype=np.float32)
    values = np.zeros((3,), dtype=np.float32)

    QueueRuntime._overwrite_central_outputs_with_opponents(
        runtime,
        actor=actor,
        batch=batch,
        obs_step=np.asarray(batch.obs, dtype=np.float32),
        actor_step=np.asarray(batch.actor, dtype=np.int64),
        logits_out=logits,
        values_out=values,
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.zeros((1, 8), dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1], dtype=np.int64))
    assert values.tolist() == [0.0, 42.0, 0.0]
    assert np.all(logits[0] == 0.0)
    assert np.all(logits[1] == -7.0)
    assert np.all(logits[2] == 0.0)
    assert actor.opponent_hidden[1].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert actor.opponent_hidden[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_overwrite_central_outputs_with_batched_opponents_groups_rows_across_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._opponent_model_locks = {"policy_000007": threading.Lock()}

    class _FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
            actor_np = actor_tensor.detach().cpu().numpy()
            self.calls.append((obs_np.copy(), actor_np.copy()))
            logits = torch.arange(obs_tensor.shape[0] * 5, dtype=torch.float32).reshape(obs_tensor.shape[0], 5)
            values = torch.arange(obs_tensor.shape[0], dtype=torch.float32) + 10.0
            return logits, values, hidden_tensor + 2.0

    model = _FakeModel()
    runtime_any._opponent_models = {"policy_000007": model}

    actor_a = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([0, 0], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray(["policy_000007", _MIRROR_OPPONENT_POLICY_ID], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    actor_b = cast(
        Any,
        SimpleNamespace(
            focal_seat_by_env=np.asarray([1, 1], dtype=np.int64),
            opponent_policy_id_by_env=np.asarray([_MIRROR_OPPONENT_POLICY_ID, "policy_000007"], dtype=object),
            opponent_hidden=torch.zeros((2, 3)),
        ),
    )
    logits_a = np.zeros((2, 5), dtype=np.float32)
    logits_b = np.zeros((2, 5), dtype=np.float32)
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((2,), dtype=np.float32)
    obs_a = np.asarray([[1.0, 0.0], [9.0, 9.0]], dtype=np.float32)
    obs_b = np.asarray([[8.0, 8.0], [2.0, 0.0]], dtype=np.float32)
    actor_step_a = np.asarray([1, 0], dtype=np.int64)
    actor_step_b = np.asarray([1, 0], dtype=np.int64)

    QueueRuntime._overwrite_central_outputs_with_batched_opponents(
        runtime,
        actors=[actor_a, actor_b],
        batches=[cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace())],
        obs_steps=[obs_a, obs_b],
        actor_steps=[actor_step_a, actor_step_b],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1, 0], dtype=np.int64))
    assert np.all(logits_a[0] == np.asarray([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.all(logits_b[1] == np.asarray([5, 6, 7, 8, 9], dtype=np.float32))
    assert values_a.tolist() == [10.0, 0.0]
    assert values_b.tolist() == [0.0, 11.0]
    assert actor_a.opponent_hidden[0].tolist() == [2.0, 2.0, 2.0]
    assert actor_b.opponent_hidden[1].tolist() == [2.0, 2.0, 2.0]


def test_build_runtime_config_minimal_batch_uses_one_unroll_per_actor() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(
                actor_process_count=12,
                envs_per_actor=8,
                actor_queue_capacity_unrolls=256,
            ),
            training=SimpleNamespace(
                batch_unrolls_per_update=128,
                actor_reload_interval_updates=1000,
            ),
        )
    )

    small = build_runtime_config(
        stack=stack,
        num_envs=1,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert small.actor_count == 1
    assert small.envs_per_actor == 1
    assert small.batch_unrolls_per_update == 1
    assert small.queue_capacity_unrolls == 1

    full = build_runtime_config(
        stack=stack,
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert full.actor_count == 12
    assert full.envs_per_actor == 8
    assert full.batch_unrolls_per_update == 12
    assert full.queue_capacity_unrolls == 12

    default = build_runtime_config(
        stack=stack,
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
    )
    assert default.batch_unrolls_per_update == 128
    assert default.queue_capacity_unrolls == 256


def test_concatenate_legal_actions_keeps_packed_ids_fast_path() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([0, 2, 1, 2], dtype=np.uint32),
        np.array([0, 2, 4], dtype=np.uint32),
    )
    unroll_a = replace(_make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)
    unroll_b = replace(_make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)

    combined = _concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.offsets.tolist() == [0, 2, 4]
    assert combined.ids.tolist() == [0, 2, 0, 2]


def test_concatenate_legal_actions_reorders_packed_rows_to_match_time_major_batch_layout() -> None:
    packed_a = LegalActionBatch.from_packed(
        np.array([10, 11, 20, 21], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    packed_b = LegalActionBatch.from_packed(
        np.array([30, 31, 40, 41], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_a,
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_b,
    )

    combined = _concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert combined.ids.tolist() == [10, 11, 30, 31, 20, 21, 40, 41]


def test_gae_advantages_matches_manual_discounted_deltas() -> None:
    rewards = np.asarray([[1.0], [0.5]], dtype=np.float32)
    values = np.asarray([[0.2], [0.3]], dtype=np.float32)
    discounts = np.asarray([[1.0], [0.0]], dtype=np.float32)
    bootstrap = np.asarray([0.4], dtype=np.float32)

    advantages = _gae_advantages(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        discounts=discounts,
        gae_lambda=0.95,
    )

    expected_last = 0.5 - 0.3
    expected_first = (1.0 + 0.3 - 0.2) + (0.95 * expected_last)
    assert advantages[:, 0].tolist() == pytest.approx([expected_first, expected_last])


def test_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        bootstrap_value=np.zeros((1,), dtype=np.float32),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_learner_batch(
        runtime,
        [unroll],
        gamma=0.99,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])


def test_build_ppo_batch_does_not_double_apply_truncation_reward() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.action_dim = 2
    runtime_any._bootstrap_values = lambda unroll: np.zeros((1,), dtype=np.float32)
    unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((2, 1, 2), dtype=np.bool_)),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
    )

    batch = QueueRuntime._build_ppo_batch(
        runtime,
        [unroll],
        gamma=0.99,
        gae_lambda=0.95,
        truncation_reward=-0.25,
        truncation_bootstrap_value=False,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert batch["discounts"][:, 0].tolist() == pytest.approx([0.99, 0.0])


def test_runtime_metrics_report_window_and_cumulative_env_step_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._runtime_start = 100.0
    runtime_any._runtime_last_metrics_time = 108.0
    runtime_any._runtime_cumulative_env_steps = 128
    runtime_any._last_published_snapshot_version = 5
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 3
    runtime_any._pfsp_pool_size = 3
    runtime_any._pfsp_quarantined_opponents = 1
    runtime_any._pfsp_champion_pool_size = 1
    runtime_any._pfsp_recent_pool_size = 1
    runtime_any._pfsp_hard_negative_pool_size = 1
    runtime_any._pfsp_last_sampled_envs = 2
    runtime_any._pfsp_last_mirror_envs = 6
    runtime_any._pfsp_last_heuristic_public_envs = 2
    runtime_any._pfsp_last_noleague_baseline_envs = 1
    runtime_any._pfsp_last_champion_envs = 1
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 1
    runtime_any._pfsp_epoch = 3

    monkeypatch.setattr("weiss_rl.runtime.time.time", lambda: 110.0)
    metrics = QueueRuntime._runtime_metrics(
        runtime,
        [
            _make_runtime_unroll(
                actor_id=0,
                unroll_seq=0,
                behavior_policy_version=4,
                counters={
                    "engine_fault_done_rows": 2,
                    "no_progress_timeout_rows": 1,
                    "pass_actions": 3,
                    "main_move_actions": 4,
                    "max_consecutive_main_moves": 2,
                },
            ),
            replace(
                _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=5),
                obs=np.zeros((2, 3, 1), dtype=np.float32),
            ),
        ],
        occupancy_samples=[0.25, 0.75],
    )

    assert metrics["batch_env_steps"] == pytest.approx(7.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(3.5)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(13.5)
    assert metrics["policy_version_lag_p50"] == pytest.approx(0.5)
    assert metrics["league_effective_update"] == pytest.approx(3.0)
    assert metrics["league_update_lag"] == pytest.approx(2.0)
    assert metrics["pfsp_quarantined_opponents"] == pytest.approx(1.0)
    assert metrics["pfsp_champion_pool_size"] == pytest.approx(1.0)
    assert metrics["pfsp_heuristic_public_envs"] == pytest.approx(2.0)
    assert metrics["pfsp_noleague_baseline_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_hard_negative_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_epoch"] == pytest.approx(3.0)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.5)
    assert metrics["collector_engine_fault_done_rows"] == pytest.approx(2.0)
    assert metrics["collector_no_progress_timeout_rows"] == pytest.approx(1.0)
    assert metrics["collector_pass_actions"] == pytest.approx(3.0)
    assert metrics["collector_main_move_actions"] == pytest.approx(4.0)
    assert metrics["collector_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert runtime_any._runtime_last_metrics_time == pytest.approx(110.0)
    assert runtime_any._runtime_cumulative_env_steps == 135


def test_resolve_actor_topology_keeps_ordered_runtime_strict_layout() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_ordered",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 12
    assert envs_per_actor == 8


def test_resolve_actor_topology_prefers_fatter_async_collectors() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 48


def test_resolve_actor_topology_prefers_64_envs_per_actor_when_available() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=128,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 64


def test_resolve_actor_topology_prefers_6x64_over_8x48_for_384_envs() -> None:
    actor_count, envs_per_actor = _resolve_actor_topology(
        num_envs=384,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 6
    assert envs_per_actor == 64


def test_runtime_honors_non_cpu_actor_device_and_disables_process_collectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(Any, SimpleNamespace(actor_id=actor_id, env=SimpleNamespace(close=lambda: None))),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cuda:0", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=True),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, object()),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._device == torch.device("cuda:0")
        assert runtime_any._actor_amp_enabled is True
        assert runtime_any._use_process_collectors is False
    finally:
        runtime.close()


def test_runtime_uses_central_batched_collection_for_typed_cpu_async(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4)),
                opponent_hidden=torch.zeros((64, 4)),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=False, compile_learner=False),
                    experiment=SimpleNamespace(role="baseline_noleague"),
                    league=None,
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_runtime_keeps_central_batched_collection_for_typed_cpu_async_league(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("weiss_rl.runtime.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        QueueRuntime,
        "_build_actor_state",
        lambda self, *, model, actor_id: cast(
            Any,
            SimpleNamespace(
                actor_id=actor_id,
                env=SimpleNamespace(close=lambda: None),
                model=model,
                compiled_model=None,
                opponent_policy_id_by_env=np.full((64,), "latest_policy_mirror", dtype=object),
                seat_hidden=torch.zeros((64, 4)),
                opponent_hidden=torch.zeros((64, 4)),
                current_batch=SimpleNamespace(obs=np.zeros((64, 8), dtype=np.float32)),
                layout_name="i16_legal_ids",
                focal_seat_by_env=np.zeros((64,), dtype=np.int64),
                rng=np.random.default_rng(7),
                snapshot_version=0,
                next_unroll_seq=0,
            ),
        ),
    )
    monkeypatch.setattr(QueueRuntime, "refresh_opponent_pool", lambda self: None)

    dummy_model = torch.nn.Linear(8, 4)
    runtime = QueueRuntime(
        stack=cast(
            Any,
            SimpleNamespace(
                config=SimpleNamespace(
                    system=SimpleNamespace(actor_device="cpu", actor_torch_threads=1),
                    training=SimpleNamespace(mixed_precision=False, compile_learner=False),
                    experiment=SimpleNamespace(role="main"),
                    league=SimpleNamespace(enabled=True, pfsp_window_episodes=50_000),
                    model=SimpleNamespace(encoder_kind="typed_v1"),
                )
            ),
        ),
        config=QueueRuntimeConfig(
            mode="train_async_fast",
            actor_count=2,
            envs_per_actor=64,
            unroll_length=32,
            batch_unrolls_per_update=96,
            queue_capacity_unrolls=256,
            profile="fast",
            base_seed=7,
            pass_action_id=51,
            actor_reload_interval_updates=1000,
        ),
        model=cast(Any, dummy_model),
        observation_dim=8,
        action_dim=16,
        run_dir=tmp_path / "league_run",
    )
    try:
        runtime_any = cast(Any, runtime)
        assert runtime_any._league_enabled is True
        assert runtime_any._use_central_batched_collection is True
        assert runtime_any._use_process_collectors is False
        assert runtime_any._collector_executor is None
    finally:
        runtime.close()


def test_fill_pending_unrolls_uses_parallel_executor_for_distinct_actors() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=4,
        envs_per_actor=1,
        unroll_length=1,
        batch_unrolls_per_update=4,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._actors = [
        cast(Any, SimpleNamespace(actor_id=actor_id))
        for actor_id in range(4)
    ]
    runtime_any._collector_result_queue = None
    runtime_any._pending_unrolls = deque()
    runtime_any._next_actor_index = 0
    runtime_any._use_central_batched_collection = False

    submitted_actor_ids: list[int] = []

    class _ImmediateFuture:
        def __init__(self, value: RuntimeUnroll) -> None:
            self._value = value

        def result(self) -> RuntimeUnroll:
            return self._value

    class _FakeExecutor:
        def submit(self, fn, actor):
            submitted_actor_ids.append(int(actor.actor_id))
            return _ImmediateFuture(fn(actor))

    runtime_any._collector_executor = _FakeExecutor()
    runtime_any._collect_actor_unroll = lambda actor: _make_runtime_unroll(
        actor_id=int(actor.actor_id),
        unroll_seq=0,
        behavior_policy_version=0,
    )

    occupancy_samples: list[float] = []
    QueueRuntime._fill_pending_unrolls(runtime, target_count=4, occupancy_samples=occupancy_samples)

    assert submitted_actor_ids == [0, 1, 2, 3]
    assert [item.actor_id for item in runtime_any._pending_unrolls] == [0, 1, 2, 3]
    assert occupancy_samples == [0.0]


def test_shared_collector_slot_round_trip_preserves_packed_unroll_payload() -> None:
    slot_config = _create_shared_collector_slot_config(
        actor_id=0,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )
    slot = _open_shared_collector_slot(slot_config, create=True)
    try:
        packed = LegalActionBatch.from_packed(
            np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
            np.array([0, 2, 3, 5, 6], dtype=np.uint32),
        )
        unroll = RuntimeUnroll(
            actor_id=0,
            unroll_seq=7,
            behavior_policy_version=11,
            unroll_hash="roundtrip",
            obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
            rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
            truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
            to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
            behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
            legal_actions=packed,
            bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
            bootstrap_actor=np.array([0, 1], dtype=np.int64),
            bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
            initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
            episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
            policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
            behavior_logits=None,
        )

        _write_unroll_to_shared_slot(slot, unroll)
        restored = _read_unroll_from_shared_slot(slot, _shared_unroll_metadata(unroll))

        assert restored.actor_id == unroll.actor_id
        assert restored.unroll_seq == unroll.unroll_seq
        assert restored.behavior_policy_version == unroll.behavior_policy_version
        assert np.array_equal(restored.obs, unroll.obs)
        assert np.array_equal(restored.actions, unroll.actions)
        assert np.array_equal(restored.bootstrap_obs, unroll.bootstrap_obs)
        assert np.array_equal(restored.bootstrap_value, unroll.bootstrap_value)
        assert np.array_equal(restored.final_hidden_state, unroll.final_hidden_state)
        assert restored.legal_actions.ids is not None
        assert restored.legal_actions.offsets is not None
        assert restored.legal_actions.ids.tolist() == unroll.legal_actions.ids.tolist()
        assert restored.legal_actions.offsets.tolist() == unroll.legal_actions.offsets.tolist()
    finally:
        slot.close(unlink=True)
