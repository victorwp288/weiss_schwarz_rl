"""Shared state for eval policy resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.eval.b1_policy_resolution import resolve_b1_policy
from weiss_rl.eval.policies.types import ResolvedEvalPolicy
from weiss_rl.eval.snapshot_policy_resolution import resolve_snapshot_registry_policy
from weiss_rl.eval.snapshot_registry_resolution import SnapshotRegistrySource
from weiss_rl.league.registry import SnapshotMeta, SnapshotRegistry


@dataclass(slots=True)
class EvalPolicyResolutionContext:
    run_dir: Path
    policy_ids: list[str]
    registry_source: SnapshotRegistrySource
    _registry_run_dir: Path | None = field(default=None, init=False, repr=False)

    @classmethod
    def load(
        cls,
        *,
        run_dir: Path,
        policy_ids: list[str],
        snapshot_registry_path: Path | None,
    ) -> EvalPolicyResolutionContext:
        registry_path = snapshot_registry_path or (
            ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / "registry.json"
        )
        return cls(
            run_dir=run_dir,
            policy_ids=policy_ids,
            registry_source=SnapshotRegistrySource.load(registry_path),
        )

    @property
    def registry_path(self) -> Path:
        return self.registry_source.path

    @property
    def registry(self) -> SnapshotRegistry:
        return self.registry_source.registry

    @property
    def snapshots_by_policy_id(self) -> dict[str, SnapshotMeta]:
        return self.registry_source.snapshots_by_policy_id

    @property
    def registry_run_dir_if_resolved(self) -> Path | None:
        return self._registry_run_dir

    def registry_run_dir_for_b1(self, *, b1_baseline_run_dir: Path | None) -> Path | None:
        if self._registry_run_dir is not None:
            return self._registry_run_dir
        if b1_baseline_run_dir is not None:
            return None
        return self.require_registry_run_dir()

    def require_registry_run_dir(self) -> Path:
        if self._registry_run_dir is None:
            self._registry_run_dir = self.registry_source.resolve_run_dir(
                run_dir=self.run_dir,
                policy_ids=self.policy_ids,
            )
        return self._registry_run_dir

    def resolve_b1_baseline_policy(
        self,
        *,
        b1_baseline_run_dir: Path | None,
        stack: StackConfig,
        observation_dim: int,
        action_dim: int,
        spec_bundle: Mapping[str, object] | None,
    ) -> ResolvedEvalPolicy:
        return resolve_b1_policy(
            run_dir=self.run_dir,
            registry_run_dir=self.registry_run_dir_for_b1(b1_baseline_run_dir=b1_baseline_run_dir),
            b1_baseline_run_dir=b1_baseline_run_dir,
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            spec_bundle=spec_bundle,
        )

    def resolve_registry_policy(
        self,
        *,
        policy_id: str,
        stack: StackConfig,
        observation_dim: int,
        action_dim: int,
        spec_bundle: Mapping[str, object] | None,
    ) -> ResolvedEvalPolicy:
        return resolve_snapshot_registry_policy(
            registry_source=self.registry_source,
            policy_id=policy_id,
            snapshot_run_dir=self.require_registry_run_dir(),
            stack=stack,
            observation_dim=observation_dim,
            action_dim=action_dim,
            spec_bundle=spec_bundle,
        )


__all__ = ["EvalPolicyResolutionContext"]
