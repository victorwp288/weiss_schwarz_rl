from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.runtime import QueueRuntime


def snapshot_relative_path(policy_id: str) -> str:
    return f"training/snapshots/{policy_id}/weights.pt"


def loaded_snapshot_models(*policy_ids: str) -> dict[str, str]:
    return {policy_id: f"loaded::{snapshot_relative_path(policy_id)}" for policy_id in policy_ids}


def write_snapshot_registry(
    tmp_path: Path,
    snapshots: Sequence[tuple[str, int]],
    *,
    champions: Sequence[str] = (),
    pinned: Sequence[str] = (),
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    for index, (policy_id, update) in enumerate(snapshots, start=1):
        registry.add_snapshot(
            policy_id=policy_id,
            update=update,
            weights_sha256=f"{index % 16:x}" * 64,
            path=snapshot_relative_path(policy_id),
        )
    for policy_id in champions:
        registry.add_champion(policy_id)
    for policy_id in pinned:
        registry.pin_snapshot(policy_id)
    registry.save(registry_path)
    return run_dir, registry_path


def opponent_pool_config(
    *,
    recent_size: int = 8,
    champion_size: int = 2,
    promotion_gate_enabled: bool = False,
    sampling: Any | None = None,
    pool: Any | None = None,
) -> SimpleNamespace:
    config = SimpleNamespace(
        snapshot_pool_recent_size=recent_size,
        snapshot_pool_champion_size=champion_size,
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.2,
        promotion_gate_enabled=promotion_gate_enabled,
    )
    if promotion_gate_enabled:
        config.promotion = SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05)))
    if sampling is not None:
        config.sampling = sampling
    if pool is not None:
        config.pool = pool
    return config


def make_opponent_pool_runtime(
    registry_path: Path,
    league_config: Any,
    *,
    run_dir: Path | None = None,
    outcomes: OnlineOutcomeTracker | None = None,
    actors: Sequence[Any] = (),
    current_update: int = 0,
    effective_update: int | None = None,
    heuristic_public_reserved_envs: int = 0,
    noleague_baseline_reserved_envs: int = 0,
) -> Any:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    if run_dir is not None:
        runtime_any._run_dir = run_dir
    runtime_any._league_enabled = True
    runtime_any._registry_path = registry_path
    runtime_any._league_config = league_config
    runtime_any._outcomes = outcomes or OnlineOutcomeTracker(window_size=128)
    runtime_any._actors = list(actors)
    runtime_any._current_learner_update = current_update
    if effective_update is not None:
        runtime_any._effective_learner_update = effective_update
    runtime_any._opponent_sampler = None
    runtime_any._opponent_candidate_ids = ()
    runtime_any._opponent_models = {}
    runtime_any._opponent_model_locks = {}
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._heuristic_public_reserved_envs_per_actor = heuristic_public_reserved_envs
    runtime_any._noleague_baseline_reserved_envs_per_actor = noleague_baseline_reserved_envs
    runtime_any._pfsp_pool_size = 0
    runtime_any._pfsp_quarantined_opponents = 0
    runtime_any._pfsp_champion_pool_size = 0
    runtime_any._pfsp_recent_pool_size = 0
    runtime_any._pfsp_hard_negative_pool_size = 0
    runtime_any._load_snapshot_model = lambda path: f"loaded::{path}"
    return runtime_any


def outcomes_from_results(policy_id: str, result: str, count: int) -> OnlineOutcomeTracker:
    outcomes = OnlineOutcomeTracker(window_size=128)
    for _ in range(count):
        outcomes.update(policy_id, result)
    return outcomes
