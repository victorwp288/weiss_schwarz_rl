from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.loading import load_snapshot_eval_model
from weiss_rl.training.promotion import (
    PROMOTION_GATE_NOLEAGUE_BASELINE_NAME,
    PROMOTION_GATE_RANDOMLEGAL_NAME,
    PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,
    build_heuristic_public_policy,
    resolve_promotion_anchor_policy_ids,
    snapshot_meta_by_policy_id,
)

PeriodicDevEvalOpponent: TypeAlias = tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]
SnapshotEvalModelLoader: TypeAlias = Callable[..., PolicyValueModel]


class HeuristicPolicyBuilder(Protocol):
    def __call__(self, spec_bundle: Mapping[str, object], *, scoring_profile: str) -> HeuristicPublicPolicy: ...


def _build_default_heuristic_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
) -> HeuristicPublicPolicy:
    return build_heuristic_public_policy(
        spec_bundle,
        scoring_profile=scoring_profile,
    )


def periodic_dev_eval_opponents(
    *,
    stack: Any,
    contract: Any,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
    load_snapshot_model: SnapshotEvalModelLoader = load_snapshot_eval_model,
    build_heuristic_policy: HeuristicPolicyBuilder | None = None,
) -> list[PeriodicDevEvalOpponent]:
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    anchor_names: list[str]
    if league is None:
        anchor_names = [PROMOTION_GATE_RANDOMLEGAL_NAME, PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = snapshot_meta_by_policy_id(registry)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    heuristic_policy_builder = build_heuristic_policy or _build_default_heuristic_policy

    opponents: list[PeriodicDevEvalOpponent] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            opponents.append((policy_id, anchor_name, None, None))
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            try:
                heuristic_policy = heuristic_policy_builder(
                    spec_bundle or {},
                    scoring_profile=heuristic_profile,
                )
            except Exception as exc:
                if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                    raise RuntimeError(
                        f"Periodic dev eval requires a heuristic-compatible simulator contract for {policy_id}"
                    ) from exc
                continue
            opponents.append((policy_id, anchor_name, None, heuristic_policy))
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        opponents.append(
            (
                policy_id,
                anchor_name,
                load_snapshot_model(
                    run_dir=run_dir,
                    snapshot_path=snapshot.path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents
