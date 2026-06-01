"""Snapshot model loading helpers for eval policy resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.loading import load_snapshot_eval_model as _shared_load_snapshot_eval_model


def load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, object] | None = None,
    spec_bundle: Mapping[str, object] | None = None,
) -> PolicyValueModel:
    return _shared_load_snapshot_eval_model(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        stack=stack,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )


def observation_spec_from_bundle(
    spec_bundle: Mapping[str, object] | None,
    *,
    run_dir: Path | None = None,
) -> Mapping[str, object] | None:
    if spec_bundle is not None:
        observation = spec_bundle.get("observation")
        if isinstance(observation, Mapping):
            return observation
    if run_dir is None:
        return None
    layout = ArtifactLayout.from_run_dir(run_dir)
    if not layout.spec_bundle_path.is_file():
        return None
    payload = json.loads(layout.spec_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"spec_bundle.json must contain an object: {layout.spec_bundle_path}")
    observation = payload.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, Mapping):
        raise RuntimeError(f"spec_bundle.json observation payload must be an object: {layout.spec_bundle_path}")
    return observation


__all__ = ["load_snapshot_eval_model", "observation_spec_from_bundle"]
