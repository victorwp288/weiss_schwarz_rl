from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.runtime import QueueRuntime

from .runtime_opponent_pool_test_support import (
    loaded_snapshot_models,
    make_opponent_pool_runtime,
    opponent_pool_config,
    write_snapshot_registry,
)


def _write_b1_anchor_registry(tmp_path: Path) -> tuple[Path, Path]:
    return write_snapshot_registry(
        tmp_path,
        [
            ("policy_000007", 7),
            ("b1_noleague_baseline", 999),
        ],
        pinned=("b1_noleague_baseline",),
    )


def test_refresh_opponent_pool_excludes_fixed_b1_anchor(tmp_path: Path) -> None:
    _run_dir, registry_path = _write_b1_anchor_registry(tmp_path)
    runtime = make_opponent_pool_runtime(registry_path, opponent_pool_config())

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_candidate_ids == ("policy_000007",)
    assert runtime._pfsp_pool_size == 1
    assert runtime._opponent_models == loaded_snapshot_models("policy_000007")


def test_refresh_opponent_pool_keeps_reserved_b1_anchor_resident(tmp_path: Path) -> None:
    _run_dir, registry_path = _write_b1_anchor_registry(tmp_path)
    runtime = make_opponent_pool_runtime(
        registry_path,
        opponent_pool_config(),
        noleague_baseline_reserved_envs=1,
    )

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_candidate_ids == ("policy_000007",)
    assert runtime._opponent_models == loaded_snapshot_models("policy_000007", "b1_noleague_baseline")


def test_refresh_opponent_pool_keeps_mixed_b1_anchor_resident(tmp_path: Path) -> None:
    _run_dir, registry_path = _write_b1_anchor_registry(tmp_path)
    sampling = SimpleNamespace(
        noleague_baseline_mix_fraction=0.15,
        noleague_baseline_mix_end_updates=-1,
    )
    runtime = make_opponent_pool_runtime(registry_path, opponent_pool_config(sampling=sampling))

    QueueRuntime.refresh_opponent_pool(runtime)

    assert runtime._opponent_candidate_ids == ("policy_000007",)
    assert runtime._opponent_models == loaded_snapshot_models("policy_000007", "b1_noleague_baseline")
