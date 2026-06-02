"""Generic snapshot-registry policy resolution for simulator eval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from weiss_rl.config import StackConfig
from weiss_rl.eval.policies.types import ResolvedEvalPolicy
from weiss_rl.eval.snapshot_model_loading import load_snapshot_eval_model, observation_spec_from_bundle
from weiss_rl.eval.snapshot_registry_resolution import SnapshotRegistrySource


def resolve_snapshot_registry_policy(
    *,
    registry_source: SnapshotRegistrySource,
    policy_id: str,
    snapshot_run_dir: Path,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    spec_bundle: Mapping[str, object] | None,
) -> ResolvedEvalPolicy:
    snapshot = registry_source.snapshot_for_policy_id(policy_id)
    if snapshot is None:
        raise FileNotFoundError(
            f"Could not resolve eval policy {policy_id!r} in snapshot registry {registry_source.path}"
        )
    model = load_snapshot_eval_model(
        run_dir=snapshot_run_dir,
        snapshot_path=snapshot.path,
        stack=stack,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec_from_bundle(spec_bundle),
        spec_bundle=spec_bundle,
    )
    return ResolvedEvalPolicy(
        policy_id=policy_id,
        kind="snapshot_registry",
        source_run_dir=snapshot_run_dir.as_posix(),
        snapshot_path=snapshot.path,
        model=model,
    )


__all__ = ["resolve_snapshot_registry_policy"]
