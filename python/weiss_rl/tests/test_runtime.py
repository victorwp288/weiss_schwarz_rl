from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.runtime import QueueRuntime, QueueRuntimeConfig, RuntimeUnroll


def _make_runtime_unroll(*, actor_id: int, unroll_seq: int, behavior_policy_version: int) -> RuntimeUnroll:
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
        logits=np.zeros((1, 1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(np.ones((1, 1, 1), dtype=np.bool_)),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
        initial_hidden_state=np.zeros((1, 1), dtype=np.float32),
        final_hidden_state=np.zeros((1, 1), dtype=np.float32),
        episode_seed=np.zeros((1, 1), dtype=np.uint64),
        policy_train_mask=np.ones((1, 1), dtype=np.bool_),
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
    )
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._pfsp_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime_any._opponent_candidate_ids == ("policy_000007",)
    assert runtime_any._pfsp_pool_size == 1
    assert runtime_any._opponent_models == {"policy_000007": "loaded::training/snapshots/policy_000007/weights.pt"}
