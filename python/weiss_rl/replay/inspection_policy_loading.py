"""Policy, registry, and simulator-spec loading for replay inspection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig, compute_config_hash256
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotMeta, SnapshotRegistry
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat


@dataclass(frozen=True, slots=True)
class LoadedReplayPolicy:
    spec: str
    label: str
    kind: str
    weights_path: Path | None
    model: PolicyValueModel | None = None
    heuristic_policy: HeuristicPublicPolicy | None = None


def resolve_registry(
    *,
    run_dir: Path | None,
    snapshot_registry_path: Path | None,
) -> tuple[Path | None, Path | None, SnapshotRegistry | None]:
    resolved_registry_path = None if snapshot_registry_path is None else snapshot_registry_path.resolve()
    resolved_run_dir = None if run_dir is None else run_dir.resolve()

    if resolved_registry_path is None and resolved_run_dir is not None:
        candidate = resolved_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
        if candidate.is_file():
            resolved_registry_path = candidate

    if resolved_run_dir is None and resolved_registry_path is not None:
        registry_path_parts = resolved_registry_path.parts[-3:]
        if registry_path_parts == ("training", "snapshots", REGISTRY_FILENAME):
            resolved_run_dir = resolved_registry_path.parents[2]

    registry = None if resolved_registry_path is None else SnapshotRegistry.load(resolved_registry_path)
    return resolved_registry_path, resolved_run_dir, registry


def opponent_context_index_for_policy(
    *,
    policy: LoadedReplayPolicy,
    opponent_context_policy_id: str | None,
    require_nonzero: bool,
) -> int | None:
    if policy.model is None or opponent_context_policy_id is None:
        return None
    context_index = int(policy.model.opponent_context_indices_for_policy_ids([opponent_context_policy_id])[0])
    if require_nonzero and context_index == 0:
        raise RuntimeError(
            f"Replay policy {policy.label!r} has no opponent-context index for {opponent_context_policy_id!r}"
        )
    return context_index


def load_policy(
    *,
    spec: str,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    run_dir: Path | None,
    registry: SnapshotRegistry | None,
    run_spec_bundle: dict[str, Any] | None,
    extra_accepted_config_hashes: set[str],
) -> LoadedReplayPolicy:
    normalized_spec = str(spec).strip()
    heuristic_profile = heuristic_public_profile_name_for_policy_id(normalized_spec)
    if heuristic_profile is not None:
        if run_spec_bundle is None:
            raise RuntimeError("Resolving heuristic-public replay policies requires spec_bundle.json in run_dir")
        return LoadedReplayPolicy(
            spec=normalized_spec,
            label=normalized_spec,
            kind="heuristic_public",
            weights_path=None,
            heuristic_policy=HeuristicPublicPolicy.from_spec_bundle(
                run_spec_bundle,
                scoring_profile=heuristic_profile,
            ),
        )

    weights_path, label = resolve_policy_weights_path(spec=spec, run_dir=run_dir, registry=registry)
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {weights_path}")

    accepted_config_hashes = accepted_snapshot_config_hashes(stack=stack, run_dir=run_dir)
    accepted_config_hashes.update(extra_accepted_config_hashes)
    observed_config_hash256 = str(payload.get("config_hash256", "")).strip()
    if observed_config_hash256 and observed_config_hash256 not in accepted_config_hashes:
        accepted_text = ", ".join(sorted(accepted_config_hashes))
        raise RuntimeError(
            f"Snapshot config hash mismatch for {weights_path}: "
            f"accepted one of [{accepted_text}], observed {observed_config_hash256}"
        )

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=load_run_observation_spec(run_spec_bundle),
        spec_bundle=run_spec_bundle,
    ).to(torch.device("cpu"))
    load_model_state_dict_with_context_compat(
        model,
        model_state_dict,
        context=f"replay snapshot {weights_path}",
    )
    model.eval()
    return LoadedReplayPolicy(
        spec=str(spec),
        label=label,
        kind="model",
        weights_path=weights_path,
        model=model,
    )


def load_run_spec_bundle(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    layout = ArtifactLayout.from_run_dir(run_dir)
    if not layout.spec_bundle_path.is_file():
        return None
    payload = json.loads(layout.spec_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"spec_bundle.json must contain an object: {layout.spec_bundle_path}")
    return dict(payload)


def accepted_snapshot_config_hashes(*, stack: StackConfig, run_dir: Path | None) -> set[str]:
    accepted_hashes = {compute_config_hash256(stack)}
    run_manifest_hash = load_run_manifest_config_hash(run_dir)
    if run_manifest_hash is not None:
        accepted_hashes.add(run_manifest_hash)
    return accepted_hashes


def normalize_config_hashes(values: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        if candidate:
            normalized.add(candidate)
    return normalized


def load_run_manifest_config_hash(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    layout = ArtifactLayout.from_run_dir(run_dir)
    if not layout.manifest_path.is_file():
        return None
    payload = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest.json must contain an object: {layout.manifest_path}")
    config_hash256 = payload.get("config_hash256")
    if isinstance(config_hash256, str) and config_hash256.strip():
        return config_hash256.strip()
    return None


def load_run_observation_spec(spec_bundle: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if spec_bundle is None:
        return None
    observation = spec_bundle.get("observation")
    if observation is None:
        return None
    if not isinstance(observation, dict):
        raise RuntimeError("spec_bundle.json observation payload must be an object")
    return dict(observation)


def load_action_catalog(spec_bundle: Mapping[str, Any] | None) -> ActionCatalog | None:
    if spec_bundle is None:
        return None
    try:
        return ActionCatalog.from_spec_bundle(spec_bundle)
    except (KeyError, TypeError, ValueError):
        return None


def resolve_policy_weights_path(
    *,
    spec: str,
    run_dir: Path | None,
    registry: SnapshotRegistry | None,
) -> tuple[Path, str]:
    normalized_spec = str(spec).strip()
    if not normalized_spec:
        raise ValueError("policy spec must be non-empty")

    spec_path = Path(normalized_spec)
    direct_candidates: list[Path] = []
    if spec_path.is_absolute():
        direct_candidates.append(spec_path)
    else:
        if run_dir is not None:
            direct_candidates.append(run_dir / spec_path)
        direct_candidates.append(spec_path)
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve(), normalized_spec

    if registry is None:
        raise RuntimeError(
            "Could not resolve policy spec "
            f"{normalized_spec!r} as a weights path, and no snapshot registry is available"
        )
    if run_dir is None:
        raise RuntimeError(f"Cannot resolve policy id {normalized_spec!r} without a run_dir")

    snapshot_meta = snapshot_by_policy_id_or_imported_seed_suffix(
        registry=registry,
        policy_id=normalized_spec,
    )
    if snapshot_meta is None:
        raise RuntimeError(f"Unknown policy id: {normalized_spec!r}")

    resolved_path = (run_dir / snapshot_meta.path).resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"Resolved policy weights path does not exist: {resolved_path}")
    return resolved_path, snapshot_meta.policy_id


def snapshot_by_policy_id_or_imported_seed_suffix(
    *,
    registry: SnapshotRegistry,
    policy_id: str,
) -> SnapshotMeta | None:
    normalized = str(policy_id).strip()
    for snapshot in registry.snapshots:
        if snapshot.policy_id == normalized:
            return snapshot
    if not normalized.startswith("seed_"):
        return None

    suffix = f"_{normalized}"
    matches = [
        snapshot
        for snapshot in registry.snapshots
        if str(snapshot.policy_id).startswith("seed_") and str(snapshot.policy_id).endswith(suffix)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    match_ids = ", ".join(sorted(snapshot.policy_id for snapshot in matches))
    raise RuntimeError(f"Ambiguous imported seed policy suffix {policy_id!r}; matches: {match_ids}")


__all__ = [
    "LoadedReplayPolicy",
    "load_action_catalog",
    "load_policy",
    "load_run_spec_bundle",
    "normalize_config_hashes",
    "opponent_context_index_for_policy",
    "resolve_policy_weights_path",
    "resolve_registry",
]
