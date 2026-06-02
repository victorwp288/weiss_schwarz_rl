"""B1 NoLeague baseline policy resolution for simulator eval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.eval.policies.set import LEGACY_NO_LEAGUE_POLICY_ID, NO_LEAGUE_POLICY_ID
from weiss_rl.eval.policies.types import ResolvedEvalPolicy
from weiss_rl.eval.snapshot_model_loading import load_snapshot_eval_model, observation_spec_from_bundle
from weiss_rl.eval.snapshot_registry_resolution import unique_paths
from weiss_rl.experiments.baselines import (
    config_marks_noleague_baseline as _shared_config_marks_noleague_baseline,
)
from weiss_rl.experiments.baselines import find_noleague_baseline_snapshot
from weiss_rl.league.registry import SnapshotMeta


def resolve_b1_policy(
    *,
    run_dir: Path,
    registry_run_dir: Path | None,
    b1_baseline_run_dir: Path | None,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    spec_bundle: Mapping[str, object] | None,
) -> ResolvedEvalPolicy:
    for candidate_run_dir in candidate_b1_run_dirs(
        run_dir=run_dir,
        registry_run_dir=registry_run_dir,
        b1_baseline_run_dir=b1_baseline_run_dir,
    ):
        snapshot = find_b1_snapshot(candidate_run_dir)
        if snapshot is None:
            continue
        model = load_snapshot_eval_model(
            run_dir=candidate_run_dir,
            snapshot_path=snapshot.path,
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            observation_spec=observation_spec_from_bundle(spec_bundle, run_dir=candidate_run_dir),
            spec_bundle=spec_bundle,
        )
        return ResolvedEvalPolicy(
            policy_id=NO_LEAGUE_POLICY_ID,
            kind="baseline_noleague",
            source_run_dir=candidate_run_dir.as_posix(),
            snapshot_path=snapshot.path,
            model=model,
        )
    raise FileNotFoundError(
        "Could not resolve the mandatory B1 NoLeague baseline. "
        "Pass --b1-baseline-run-dir or evaluate from a baseline_noleague run that persisted the canonical baseline snapshot."
    )


def candidate_b1_run_dirs(
    *,
    run_dir: Path,
    registry_run_dir: Path | None,
    b1_baseline_run_dir: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if b1_baseline_run_dir is not None:
        candidates.append(Path(b1_baseline_run_dir))
    if registry_run_dir is not None:
        candidates.append(Path(registry_run_dir))
    candidates.append(Path(run_dir))
    return unique_paths(candidates)


def config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    return _shared_config_marks_noleague_baseline(config_canonical)


def find_b1_snapshot(run_dir: Path) -> SnapshotMeta | None:
    return find_noleague_baseline_snapshot(
        run_dir,
        policy_id_candidates=(NO_LEAGUE_POLICY_ID, LEGACY_NO_LEAGUE_POLICY_ID),
    )


__all__ = [
    "candidate_b1_run_dirs",
    "config_marks_noleague_baseline",
    "find_b1_snapshot",
    "resolve_b1_policy",
]
