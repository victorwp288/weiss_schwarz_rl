"""Evaluation policy resolution for simulator-backed thesis eval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from weiss_rl.config import StackConfig
from weiss_rl.eval.b1_policy_resolution import (
    candidate_b1_run_dirs as _candidate_b1_run_dirs,
)
from weiss_rl.eval.b1_policy_resolution import (
    config_marks_noleague_baseline as _config_marks_noleague_baseline,
)
from weiss_rl.eval.b1_policy_resolution import (
    find_b1_snapshot as _find_b1_snapshot,
)
from weiss_rl.eval.b1_policy_resolution import (
    resolve_b1_policy as _resolve_b1_policy,
)
from weiss_rl.eval.policies.resolution_context import EvalPolicyResolutionContext
from weiss_rl.eval.policies.set import (
    NO_LEAGUE_POLICY_ID,
)
from weiss_rl.eval.policies.types import ResolvedEvalPolicy
from weiss_rl.eval.snapshot_model_loading import (
    load_snapshot_eval_model as _load_snapshot_eval_model,
)
from weiss_rl.eval.snapshot_model_loading import (
    observation_spec_from_bundle as _observation_spec_from_bundle,
)
from weiss_rl.eval.snapshot_policy_resolution import (
    resolve_snapshot_registry_policy as _resolve_snapshot_registry_policy,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    common_search_root as _common_search_root,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    is_recursive_registry_search_root as _is_recursive_registry_search_root,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    resolve_snapshot_registry_run_dir as _resolve_snapshot_registry_run_dir_impl,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    sha256_file as _sha256_file,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    should_include_common_search_root as _should_include_common_search_root,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    snapshot_by_policy_id_or_imported_seed_suffix as _snapshot_by_policy_id_or_imported_seed_suffix,
)
from weiss_rl.eval.snapshot_registry_resolution import (
    unique_paths as _unique_paths,
)
from weiss_rl.eval.static_policy_resolution import (
    resolve_static_eval_policy as _resolve_static_eval_policy,
)
from weiss_rl.league.registry import SnapshotRegistry


def resolve_eval_policies(
    *,
    stack: StackConfig,
    policy_ids: list[str],
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
    spec_bundle: Mapping[str, object] | None = None,
    snapshot_registry_path: Path | None = None,
    b1_baseline_run_dir: Path | None = None,
) -> dict[str, ResolvedEvalPolicy]:
    context = EvalPolicyResolutionContext.load(
        run_dir=run_dir,
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
    )
    resolved: dict[str, ResolvedEvalPolicy] = {}

    for policy_id in policy_ids:
        static_policy = _resolve_static_eval_policy(policy_id=policy_id, spec_bundle=spec_bundle)
        if static_policy is not None:
            resolved[policy_id] = static_policy
            continue
        if policy_id == NO_LEAGUE_POLICY_ID:
            resolved[policy_id] = context.resolve_b1_baseline_policy(
                b1_baseline_run_dir=b1_baseline_run_dir,
                stack=stack,
                observation_dim=observation_dim,
                action_dim=action_dim,
                spec_bundle=spec_bundle,
            )
            continue

        resolved[policy_id] = context.resolve_registry_policy(
            policy_id=policy_id,
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            spec_bundle=spec_bundle,
        )

    return resolved


def _resolve_snapshot_registry_run_dir(
    *,
    run_dir: Path,
    registry_path: Path,
    registry: SnapshotRegistry,
    policy_ids: list[str],
) -> Path:
    return _resolve_snapshot_registry_run_dir_impl(
        run_dir=run_dir,
        registry_path=registry_path,
        registry=registry,
        policy_ids=policy_ids,
    )


__all__ = [
    "ResolvedEvalPolicy",
    "_candidate_b1_run_dirs",
    "_common_search_root",
    "_config_marks_noleague_baseline",
    "_find_b1_snapshot",
    "_is_recursive_registry_search_root",
    "_load_snapshot_eval_model",
    "_observation_spec_from_bundle",
    "_resolve_b1_policy",
    "_resolve_snapshot_registry_policy",
    "_resolve_snapshot_registry_run_dir",
    "_resolve_static_eval_policy",
    "_sha256_file",
    "_should_include_common_search_root",
    "_snapshot_by_policy_id_or_imported_seed_suffix",
    "_unique_paths",
    "resolve_eval_policies",
]
