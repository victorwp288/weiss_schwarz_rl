"""Shared eval policy resolution types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.model import PolicyValueModel


@dataclass(frozen=True, slots=True)
class ResolvedEvalPolicy:
    policy_id: str
    kind: str
    source_run_dir: str | None = None
    snapshot_path: str | None = None
    model: PolicyValueModel | None = None
    heuristic_policy: HeuristicPublicPolicy | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": self.kind,
            "source_run_dir": self.source_run_dir,
            "snapshot_path": self.snapshot_path,
        }


__all__ = ["ResolvedEvalPolicy"]
