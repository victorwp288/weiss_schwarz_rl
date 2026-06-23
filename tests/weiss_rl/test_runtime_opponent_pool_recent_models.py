from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime import QueueRuntime

from .runtime_opponent_pool_test_support import (
    loaded_snapshot_models,
    make_opponent_pool_runtime,
    opponent_pool_config,
    write_snapshot_registry,
)


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_promotion_gate_enabled(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [
            ("policy_000007", 7),
            ("policy_000008", 8),
            ("b1_noleague_baseline", 999),
        ],
        champions=("policy_000007",),
        pinned=("b1_noleague_baseline",),
    )
    runtime = make_opponent_pool_runtime(registry_path, opponent_pool_config(promotion_gate_enabled=True))

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_champion_ids == ("policy_000007",)
    assert runtime._opponent_recent_ids == ("policy_000008",)
    assert runtime._opponent_candidate_ids == ("policy_000007", "policy_000008")
    assert runtime._pfsp_pool_size == 2
    assert runtime._pfsp_recent_pool_size == 1
    assert runtime._opponent_models == loaded_snapshot_models("policy_000007", "policy_000008")


def test_refresh_opponent_pool_uses_probationary_recent_pool_before_first_champion(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [
            ("policy_000007", 7),
            ("policy_000008", 8),
            ("b1_noleague_baseline", 999),
        ],
        pinned=("b1_noleague_baseline",),
    )
    runtime = make_opponent_pool_runtime(registry_path, opponent_pool_config(promotion_gate_enabled=True))

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_champion_ids == ()
    assert runtime._opponent_recent_ids == ("policy_000008",)
    assert runtime._opponent_candidate_ids == ("policy_000008",)
    assert runtime._pfsp_pool_size == 1
    assert runtime._pfsp_recent_pool_size == 1
    assert runtime._opponent_models == loaded_snapshot_models("policy_000008")


def test_refresh_opponent_pool_keeps_models_for_inflight_stale_assignments(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [
            ("policy_000007", 7),
            ("policy_000008", 8),
            ("b1_noleague_baseline", 999),
        ],
        pinned=("b1_noleague_baseline",),
    )
    actor = SimpleNamespace(opponent_policy_id_by_env=np.asarray(["policy_000007"], dtype=object))
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(promotion_gate_enabled=True),
        actors=(actor,),
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_candidate_ids == ("policy_000008",)
    assert runtime._opponent_models == loaded_snapshot_models("policy_000008", "policy_000007")


def test_refresh_opponent_pool_keeps_small_recent_reservoir_when_champions_exist(tmp_path: Path) -> None:
    _run_dir, registry_path = write_snapshot_registry(
        tmp_path,
        [
            ("policy_000190", 190),
            ("policy_000191", 191),
            ("policy_000192", 192),
            ("policy_000193", 193),
        ],
        champions=("policy_000190",),
    )
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(champion_size=4, promotion_gate_enabled=True),
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_champion_ids == ("policy_000190",)
    assert runtime._opponent_recent_ids == ("policy_000192", "policy_000193")
    assert runtime._opponent_candidate_ids == ("policy_000190", "policy_000192", "policy_000193")
    assert runtime._pfsp_pool_size == 3
    assert runtime._pfsp_recent_pool_size == 2
    assert runtime._opponent_models == loaded_snapshot_models("policy_000190", "policy_000192", "policy_000193")
