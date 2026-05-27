"""B1 NoLeague baseline anchor orchestration for training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from weiss_rl.config import canonical_config_dict
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.training.promotion import (
    PROMOTION_GATE_NOLEAGUE_BASELINE_NAME,
    PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
    promotion_anchor_policy_id_candidates,
)
from weiss_rl.training.snapshots import save_snapshot_registry_with_retention, sync_snapshot_registry_retention

PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"

WriteCheckpointFn = Callable[..., Any]
ImportNoLeagueBaselineAnchorFn = Callable[..., tuple[Path, str, int]]
ModelGuidancePayloadFn = Callable[[Any], dict[str, Any]]
WriteSnapshotArtifactFn = Callable[..., tuple[Path, str]]
ExperimentRoleFn = Callable[[Any], str]


def ensure_noleague_baseline_anchor(
    *,
    stack: Any,
    training_paths: Any,
    run_dir: Path,
    learner: Any,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    baseline_run_dir: Path | None = None,
    permit_current_run_alias: bool = False,
    source_checkpoint_path: Path | None = None,
    update: int | None = None,
    write_checkpoint_fn: WriteCheckpointFn,
    import_noleague_baseline_anchor_fn: ImportNoLeagueBaselineAnchorFn,
    model_guidance_payload_fn: ModelGuidancePayloadFn,
    write_snapshot_artifact_fn: WriteSnapshotArtifactFn,
    experiment_role_fn: ExperimentRoleFn,
) -> str | None:
    league = stack.config.league
    training_config = stack.config.training
    experiment_role = experiment_role_fn(stack)
    requires_anchor = bool(
        league is not None
        and league.enabled
        and league.promotion_gate_enabled
        and PROMOTION_GATE_NOLEAGUE_BASELINE_NAME in league.promotion_anchor_set_v1.required
    )
    materialize_external_anchor = baseline_run_dir is not None
    if not requires_anchor and not permit_current_run_alias and not materialize_external_anchor:
        return None
    if learner.model is None:
        raise RuntimeError("Cannot ensure the NoLeague baseline anchor without a learner model")

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    available_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    existing_policy_id = next(
        (
            candidate
            for candidate in promotion_anchor_policy_id_candidates(PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
            if candidate in available_policy_ids
        ),
        None,
    )
    if existing_policy_id is not None and baseline_run_dir is None and permit_current_run_alias:
        existing_snapshot = next(
            (snapshot for snapshot in registry.snapshots if snapshot.policy_id == existing_policy_id),
            None,
        )
        resolved_update = int(learner.update_count if update is None else update)
        if existing_snapshot is None or int(existing_snapshot.update) < resolved_update:
            existing_policy_id = None
    if existing_policy_id is not None:
        registry.pin_snapshot(existing_policy_id)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return existing_policy_id

    if baseline_run_dir is not None:
        weights_path, weights_sha256, imported_update = import_noleague_baseline_anchor_fn(
            training_paths=training_paths,
            run_dir=run_dir,
            baseline_run_dir=baseline_run_dir,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
        registry.add_snapshot(
            policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            update=int(imported_update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
        )
        registry.pin_snapshot(PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported promotion anchor: "
            f"anchor={PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
            f"policy_id={PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
            f"source_run_dir={Path(baseline_run_dir).resolve()}"
        )
        return PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID

    if not permit_current_run_alias:
        if requires_anchor:
            raise RuntimeError(
                "The canonical B1 NoLeague baseline is required for this training run. "
                "Pass --b1-baseline-run-dir pointing at a completed baseline_noleague run."
            )
        return None

    resolved_update = int(learner.update_count if update is None else update)
    checkpoint_path = (
        training_paths.checkpoints_dir / PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT
        if source_checkpoint_path is None
        else Path(source_checkpoint_path)
    )
    if source_checkpoint_path is None:
        write_checkpoint_fn(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            algorithm=str(training_config.algorithm).strip() if training_config is not None else None,
            spec_hash256=spec_hash256,
        )
    guidance_payload = model_guidance_payload_fn(learner.model)
    weights_path, weights_sha256 = write_snapshot_artifact_fn(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=learner.model.state_dict(),
        structured_policy_contract=(
            None if stack.config.model is None else stack.config.model.structured_policy_contract
        ),
        public_heuristic_logit_bias_scale=guidance_payload.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance_payload.get("public_heuristic_actor_logit_bias_scale"),
    )
    registry.add_snapshot(
        policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
    )
    registry.pin_snapshot(PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    print(
        "Persisted canonical B1 baseline alias: "
        f"anchor={PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
        f"policy_id={PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
        f"experiment_role={experiment_role or 'unknown'} update={resolved_update}"
    )
    return PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
