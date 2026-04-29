"""Snapshot import services for training runs.

This module owns compatibility-sensitive imports from completed run registries
into the current run's snapshot registry. The public ids and metadata payloads
are intentionally stable because eval policy resolution and thesis artifacts
depend on them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.training.anchor_resolution import promotion_anchor_policy_id_candidates
from weiss_rl.training.eval_schedule import is_noleague_baseline_role
from weiss_rl.training.snapshot_artifacts import (
    save_snapshot_registry_with_retention,
    sha256_file,
    sync_snapshot_registry_retention,
    write_snapshot_artifact,
)

PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
FIXED_OPPONENT_EXCLUSIONS = frozenset({PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID})


class SnapshotImportPaths(Protocol):
    snapshots_dir: Path
    checkpoints_dir: Path


@dataclass(frozen=True, slots=True)
class EnsureNoleagueBaselineAnchorResult:
    policy_id: str | None
    message: str | None = None


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def noleague_baseline_policy_id_candidates() -> tuple[str, str]:
    return (PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID, PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)


def canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    config = config_canonical.get("config")
    return config if isinstance(config, Mapping) else config_canonical


def role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    experiment = canonical_config_sections(config_canonical).get("experiment", {})
    if isinstance(experiment, Mapping):
        role = str(experiment.get("role", "")).strip()
        if role:
            return role
    return ""


def legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    training_family = canonical_config_sections(config_canonical).get("training_family_a", {})
    if isinstance(training_family, Mapping):
        return str(training_family.get("mode", "")).strip()
    return ""


def config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    role = role_from_config_canonical(config_canonical)
    if role:
        return is_noleague_baseline_role(role)
    legacy_mode = legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode:
        return legacy_mode == "b1_no_league"
    return False


def assert_noleague_baseline_config(config_canonical: Mapping[str, Any]) -> None:
    role = role_from_config_canonical(config_canonical)
    if role:
        if not is_noleague_baseline_role(role):
            raise RuntimeError(
                f"Imported B1 baseline must come from a dedicated baseline_noleague run, got experiment.role={role!r}"
            )
        return
    legacy_mode = legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode and legacy_mode != "b1_no_league":
        raise RuntimeError(
            "Imported B1 baseline must come from a dedicated baseline_noleague run, "
            f"got training_family_a.mode={legacy_mode!r}"
        )


def read_optional_hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _model_section_for_contract(section: Any) -> Any:
    if not isinstance(section, Mapping):
        return section
    normalized = dict(section)
    normalized.pop("public_heuristic_logit_bias_scale", None)
    normalized.pop("public_heuristic_actor_logit_bias_scale", None)
    normalized.pop("public_heuristic_logit_bias_start_updates", None)
    normalized.pop("public_heuristic_logit_bias_end_updates", None)
    normalized.pop("public_heuristic_logit_bias_final_scale", None)
    return normalized


def _validate_model_state_contract(
    *,
    label: str,
    source_path: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
) -> None:
    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"{label} weights payload is missing model_state_dict: {source_path}")
    source_keys = set(source_model_state_dict)
    expected_keys = set(expected_model_state_dict)
    if source_keys != expected_keys:
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        raise RuntimeError(
            f"{label} model contract does not match the current run: missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(expected_keys):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                f"{label} tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def validate_imported_snapshot_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = load_json_object(manifest_path, label="imported B1 manifest") if manifest_path.is_file() else None
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict):
        source_config_sections = canonical_config_sections(source_config_canonical)
        assert_noleague_baseline_config(source_config_canonical)
        if isinstance(expected_config_canonical, dict):
            expected_config_sections = canonical_config_sections(expected_config_canonical)
            for section_name in ("model", "environment"):
                source_section = source_config_sections.get(section_name)
                expected_section = expected_config_sections.get(section_name)
                if source_section is None or expected_section is None:
                    continue
                if section_name == "model":
                    source_section = _model_section_for_contract(source_section)
                    expected_section = _model_section_for_contract(expected_section)
                if source_section != expected_section:
                    raise RuntimeError(
                        f"Imported B1 baseline config does not match the current run for section={section_name!r}"
                    )

    if expected_spec_hash256 is not None:
        source_spec_hash = read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported B1 baseline spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    _validate_model_state_contract(
        label="Imported B1 baseline",
        source_path=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
    )


def find_noleague_baseline_snapshot(run_dir: Path) -> SnapshotMeta | None:
    layout = ArtifactLayout.from_run_dir(run_dir)
    registry_path = layout.training_snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    registry = SnapshotRegistry.load(registry_path)
    snapshots_by_id = {snapshot.policy_id: snapshot for snapshot in registry.snapshots}
    for policy_id in noleague_baseline_policy_id_candidates():
        snapshot = snapshots_by_id.get(policy_id)
        if snapshot is not None:
            return snapshot

    manifest_path = layout.manifest_path
    if not manifest_path.is_file():
        return None
    manifest = load_json_object(manifest_path, label="run manifest")
    config_canonical = manifest.get("config_canonical", {})
    if not isinstance(config_canonical, dict):
        return None
    if not config_marks_noleague_baseline(config_canonical):
        return None
    if not registry.snapshots:
        return None
    return max(registry.snapshots, key=lambda snapshot: snapshot.sort_key())


def import_noleague_baseline_anchor(
    *,
    training_paths: SnapshotImportPaths,
    run_dir: Path,
    baseline_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> tuple[Path, str, int]:
    source_run_dir = Path(baseline_run_dir).resolve()
    source_snapshot = find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    snapshot_dir = training_paths.snapshots_dir / PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )
    imported_payload = dict(payload)
    imported_payload["policy_id"] = PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
    imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
    imported_payload["imported_from_snapshot_path"] = source_snapshot.path
    torch.save(imported_payload, weights_path)
    weights_sha256 = sha256_file(weights_path)

    write_json_file(
        snapshot_dir / SNAPSHOT_METADATA_FILENAME,
        {
            "format": "imported_train_snapshot_metadata_v1",
            "policy_id": PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            "update": int(source_snapshot.update),
            "weights_path": snapshot_weights_relpath(PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID),
            "weights_sha256": weights_sha256,
            "imported_from_run_dir": source_run_dir.as_posix(),
            "imported_from_policy_id": source_snapshot.policy_id,
            "imported_from_snapshot_path": source_snapshot.path,
        },
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def ensure_noleague_baseline_anchor(
    *,
    stack: StackConfig,
    training_paths: SnapshotImportPaths,
    run_dir: Path,
    model_state_dict: dict[str, Any],
    learner_update_count: int,
    device: torch.device,
    config_hash256: str,
    expected_config_canonical: dict[str, Any],
    spec_hash256: str | None = None,
    baseline_run_dir: Path | None = None,
    permit_current_run_alias: bool = False,
    source_checkpoint_path: Path | None = None,
    update: int | None = None,
    write_checkpoint: Callable[[Path], None] | None = None,
    guidance_payload: Mapping[str, Any] | None = None,
    experiment_role: str = "",
) -> EnsureNoleagueBaselineAnchorResult:
    league = stack.config.league
    training_config = stack.config.training
    reference_policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
    if not reference_policy_id:
        reference_policy_id = PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    raw_b1_distill_enabled = bool(getattr(raw_b1_distill, "enabled", False)) and (
        float(getattr(raw_b1_distill, "coef", 0.0)) != 0.0 or float(getattr(raw_b1_distill, "final_coef", 0.0)) != 0.0
    )
    raw_b1_distill_policy_id = str(getattr(raw_b1_distill, "teacher_policy_id", "") or "").strip()
    reference_needs_b1_anchor = (
        float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0)) != 0.0
        or float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0)) != 0.0
        or float(getattr(training_config, "b1_opponent_reference_policy_top_action_bc_coef", 0.0)) != 0.0
        or raw_b1_distill_enabled
    ) and reference_policy_id in promotion_anchor_policy_id_candidates(PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
    if raw_b1_distill_enabled and raw_b1_distill_policy_id:
        reference_needs_b1_anchor = reference_needs_b1_anchor or (
            raw_b1_distill_policy_id in promotion_anchor_policy_id_candidates(PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
        )
    requires_anchor = bool(
        reference_needs_b1_anchor
        or (
            league is not None
            and league.enabled
            and league.promotion_gate_enabled
            and PROMOTION_GATE_NOLEAGUE_BASELINE_NAME in league.promotion_anchor_set_v1.required
        )
    )
    if not requires_anchor and not permit_current_run_alias:
        return EnsureNoleagueBaselineAnchorResult(policy_id=None)

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
        resolved_update = int(learner_update_count if update is None else update)
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
        return EnsureNoleagueBaselineAnchorResult(policy_id=existing_policy_id)

    if baseline_run_dir is not None:
        weights_path, weights_sha256, imported_update = import_noleague_baseline_anchor(
            training_paths=training_paths,
            run_dir=run_dir,
            baseline_run_dir=baseline_run_dir,
            expected_model_state_dict=model_state_dict,
            expected_config_canonical=expected_config_canonical,
            expected_spec_hash256=spec_hash256,
        )
        registry.add_snapshot(
            policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            update=int(imported_update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="baseline_anchor",
        )
        registry.pin_snapshot(PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        return EnsureNoleagueBaselineAnchorResult(
            policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
            message=(
                "Imported promotion anchor: "
                f"anchor={PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
                f"policy_id={PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
                f"source_run_dir={Path(baseline_run_dir).resolve()}"
            ),
        )

    if not permit_current_run_alias:
        if requires_anchor:
            raise RuntimeError(
                "The canonical B1 NoLeague baseline is required for this training run. "
                "Pass --b1-baseline-run-dir pointing at a completed baseline_noleague run."
            )
        return EnsureNoleagueBaselineAnchorResult(policy_id=None)

    resolved_update = int(learner_update_count if update is None else update)
    checkpoint_path = (
        training_paths.checkpoints_dir / PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT
        if source_checkpoint_path is None
        else Path(source_checkpoint_path)
    )
    if source_checkpoint_path is None:
        if write_checkpoint is None:
            raise RuntimeError("write_checkpoint callback is required when persisting a current-run baseline alias")
        write_checkpoint(checkpoint_path)
    guidance = guidance_payload or {}
    weights_path, weights_sha256 = write_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        config_hash256=config_hash256,
        device=device,
        model_state_dict=model_state_dict,
        public_heuristic_logit_bias_scale=guidance.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance.get("public_heuristic_actor_logit_bias_scale"),
    )
    registry.add_snapshot(
        policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=resolved_update,
        weights_sha256=weights_sha256,
        path=weights_path.relative_to(run_dir).as_posix(),
        source_kind="baseline_anchor",
    )
    registry.pin_snapshot(PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID)
    save_snapshot_registry_with_retention(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        registry=registry,
    )
    return EnsureNoleagueBaselineAnchorResult(
        policy_id=PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        message=(
            "Persisted canonical B1 baseline alias: "
            f"anchor={PROMOTION_GATE_NOLEAGUE_BASELINE_NAME} "
            f"policy_id={PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID} "
            f"experiment_role={experiment_role or 'unknown'} update={resolved_update}"
        ),
    )


def validate_seed_snapshot_import_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = (
        load_json_object(manifest_path, label="seed snapshot manifest") if manifest_path.is_file() else None
    )
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict) and isinstance(expected_config_canonical, dict):
        source_config_sections = canonical_config_sections(source_config_canonical)
        expected_config_sections = canonical_config_sections(expected_config_canonical)
        for section_name in ("model", "environment"):
            source_section = source_config_sections.get(section_name)
            expected_section = expected_config_sections.get(section_name)
            if source_section is None or expected_section is None:
                continue
            if section_name == "model":
                source_section = _model_section_for_contract(source_section)
                expected_section = _model_section_for_contract(expected_section)
            if source_section != expected_section:
                raise RuntimeError(
                    f"Imported seed snapshot config does not match the current run for section={section_name!r}"
                )

    if expected_spec_hash256 is not None:
        source_spec_hash = read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported seed snapshot spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    _validate_model_state_contract(
        label="Imported seed snapshot",
        source_path=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
    )


def validate_snapshot_tensor_contract(
    *,
    label: str,
    source_path: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
) -> None:
    _validate_model_state_contract(
        label=label,
        source_path=source_path,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
    )


def seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    source_hash = hashlib.sha1(source_run_dir.as_posix().encode("utf-8")).hexdigest()[:10]
    safe_policy_id = str(source_policy_id).replace("/", "_").replace("\\", "_").strip()
    return f"seed_{source_hash}_{safe_policy_id}"


def import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: SnapshotImportPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    max_update: int | None = None,
    exclude_source_policy_ids: Sequence[str] = (),
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    source_run_dir = Path(seed_snapshot_run_dir).resolve()
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve a snapshot registry in the seed snapshot run: {source_registry_path}"
        )
    source_registry = SnapshotRegistry.load(source_registry_path)
    excluded_source_policy_ids = {str(policy_id).strip() for policy_id in exclude_source_policy_ids}
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if snapshot.policy_id not in noleague_baseline_policy_id_candidates()
        and snapshot.policy_id not in excluded_source_policy_ids
        and (max_update is None or int(snapshot.update) <= int(max_update))
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        imported_policy_id = seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )
        if imported_policy_id in existing_policy_ids:
            imported_policy_ids.append(imported_policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"Resolved seed snapshot is missing its weights artifact: {source_weights_path}")
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Imported seed snapshot weights payload must be a dict: {source_weights_path}")
        validate_seed_snapshot_import_contract(
            source_run_dir=source_run_dir,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
            expected_config_canonical=expected_config_canonical,
            expected_spec_hash256=expected_spec_hash256,
        )
        snapshot_dir = training_paths.snapshots_dir / imported_policy_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = imported_policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["seeded_from_external_registry"] = True
        imported_payload["source_was_champion"] = source_snapshot.policy_id in source_champions
        torch.save(imported_payload, weights_path)
        weights_sha256 = sha256_file(weights_path)
        write_json_file(
            snapshot_dir / SNAPSHOT_METADATA_FILENAME,
            {
                "format": "seeded_train_snapshot_metadata_v1",
                "policy_id": imported_policy_id,
                "update": int(source_snapshot.update),
                "weights_path": snapshot_weights_relpath(imported_policy_id),
                "weights_sha256": weights_sha256,
                "imported_from_run_dir": source_run_dir.as_posix(),
                "imported_from_policy_id": source_snapshot.policy_id,
                "imported_from_snapshot_path": source_snapshot.path,
                "source_was_champion": source_snapshot.policy_id in source_champions,
            },
        )
        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="seed_import",
        )
        existing_policy_ids.add(imported_policy_id)
        imported_policy_ids.append(imported_policy_id)

    if imported_policy_ids:
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported seeded snapshot pool: "
            f"count={len(imported_policy_ids)} "
            f"source_run_dir={source_run_dir.as_posix()}"
        )
    return imported_policy_ids


def source_snapshot_is_resume_league_snapshot(snapshot: SnapshotMeta, *, rejected_policy_ids: set[str]) -> bool:
    policy_id = str(snapshot.policy_id).strip()
    if (
        not policy_id
        or policy_id in rejected_policy_ids
        or policy_id in FIXED_OPPONENT_EXCLUSIONS
        or policy_id.startswith("seed_")
    ):
        return False
    source_kind = str(getattr(snapshot, "source_kind", "local") or "local").strip().lower()
    return source_kind not in {"seed_import", "baseline_anchor"}


def validate_existing_resume_league_import(
    *,
    training_paths: SnapshotImportPaths,
    source_run_dir: Path,
    source_snapshot: SnapshotMeta,
) -> None:
    policy_id = str(source_snapshot.policy_id)
    metadata_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise RuntimeError(f"Existing resume league snapshot import is missing metadata: {metadata_path}")
    metadata = load_json_object(metadata_path, label="existing resume league snapshot metadata")
    expected_source_run_dir = source_run_dir.as_posix()
    if (
        metadata.get("format") != "resume_league_snapshot_metadata_v1"
        or str(metadata.get("imported_from_run_dir", "")) != expected_source_run_dir
        or str(metadata.get("imported_from_policy_id", "")) != policy_id
        or str(metadata.get("imported_from_snapshot_path", "")) != str(source_snapshot.path)
    ):
        raise RuntimeError(
            "Existing resume league snapshot policy_id collision does not match the requested import: "
            f"policy_id={policy_id} metadata_path={metadata_path}"
        )


def infer_run_dir_from_checkpoint_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    resolved = Path(checkpoint_path).resolve()
    checkpoint_dir = resolved.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    return training_dir.parent


def import_resume_league_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: SnapshotImportPaths,
    run_dir: Path,
    resume_checkpoint_path: Path,
    max_update: int,
    expected_model_state_dict: dict[str, Any],
) -> list[str]:
    source_run_dir = infer_run_dir_from_checkpoint_path(resume_checkpoint_path)
    if source_run_dir is None or source_run_dir.resolve() == Path(run_dir).resolve():
        return []
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        return []
    source_registry = SnapshotRegistry.load(source_registry_path)
    rejected_policy_ids = set(getattr(source_registry, "rejected_snapshots", ()))
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if int(snapshot.update) <= int(max_update)
        and source_snapshot_is_resume_league_snapshot(snapshot, rejected_policy_ids=rejected_policy_ids)
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    source_champions = set(source_registry.champion_snapshots)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        policy_id = str(source_snapshot.policy_id)
        if policy_id in existing_policy_ids:
            validate_existing_resume_league_import(
                training_paths=training_paths,
                source_run_dir=source_run_dir,
                source_snapshot=source_snapshot,
            )
            imported_policy_ids.append(policy_id)
            continue
        source_weights_path = source_run_dir / source_snapshot.path
        if not source_weights_path.is_file():
            raise FileNotFoundError(f"Resolved resume league snapshot is missing weights: {source_weights_path}")
        payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Imported resume league snapshot weights payload must be a dict: {source_weights_path}")
        validate_snapshot_tensor_contract(
            label="Imported resume league snapshot",
            source_path=source_weights_path,
            payload=payload,
            expected_model_state_dict=expected_model_state_dict,
        )
        snapshot_dir = training_paths.snapshots_dir / policy_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        weights_path = snapshot_dir / SNAPSHOT_WEIGHTS_FILENAME
        imported_payload = dict(payload)
        imported_payload["policy_id"] = policy_id
        imported_payload["imported_from_run_dir"] = source_run_dir.as_posix()
        imported_payload["imported_from_policy_id"] = source_snapshot.policy_id
        imported_payload["imported_from_snapshot_path"] = source_snapshot.path
        imported_payload["resumed_from_league_registry"] = True
        torch.save(imported_payload, weights_path)
        weights_sha256 = sha256_file(weights_path)
        write_json_file(
            snapshot_dir / SNAPSHOT_METADATA_FILENAME,
            {
                "format": "resume_league_snapshot_metadata_v1",
                "policy_id": policy_id,
                "update": int(source_snapshot.update),
                "weights_path": snapshot_weights_relpath(policy_id),
                "weights_sha256": weights_sha256,
                "imported_from_run_dir": source_run_dir.as_posix(),
                "imported_from_policy_id": source_snapshot.policy_id,
                "imported_from_snapshot_path": source_snapshot.path,
                "source_kind": str(getattr(source_snapshot, "source_kind", "local") or "local"),
            },
        )
        registry.add_snapshot(
            policy_id=policy_id,
            update=int(source_snapshot.update),
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
            source_kind="league_import",
        )
        if source_snapshot.policy_id in source_champions:
            registry.add_champion(policy_id)
        existing_policy_ids.add(policy_id)
        imported_policy_ids.append(policy_id)

    if imported_policy_ids:
        save_snapshot_registry_with_retention(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            registry=registry,
        )
        print(
            "Imported resume league snapshot pool: "
            f"count={len(imported_policy_ids)} "
            f"source_run_dir={source_run_dir.as_posix()}"
        )
    return imported_policy_ids
