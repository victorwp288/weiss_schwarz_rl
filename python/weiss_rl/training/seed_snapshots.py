"""Seed snapshot pool import helpers for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.models.state_dict_compat import state_dict_key_mismatch_for_context_compat
from weiss_rl.training.import_contracts import (
    canonical_config_sections,
    config_sections_match_for_import_contract,
    read_optional_hash_file,
)
from weiss_rl.training.promotion import PROMOTION_GATE_NOLEAGUE_BASELINE_NAME, promotion_anchor_policy_id_candidates
from weiss_rl.training.run_metadata import load_json_object
from weiss_rl.training.snapshots import (
    save_snapshot_registry_with_retention,
    sync_snapshot_registry_retention,
    write_imported_snapshot_artifact,
)
from weiss_rl.training.snapshots import (
    seed_snapshot_policy_id as _seed_snapshot_policy_id,
)

ALLOWED_SEED_SNAPSHOT_SOURCE_ROLES = frozenset(
    {"main", "guided_league_seed", "guided_league_bootstrap", "ablation_guided"}
)
SEED_SNAPSHOT_CHAMPION_IMPORT_POLICIES = frozenset({"source_champions", "pinned", "all", "none"})
SEED_SNAPSHOT_IMPORT_FILTERS = frozenset({"source_champions", "pinned", "pinned_or_source_champions", "all", "none"})


def _source_experiment_role(config_canonical: dict[str, Any]) -> str:
    experiment = canonical_config_sections(config_canonical).get("experiment", {})
    if isinstance(experiment, dict):
        return str(experiment.get("role", "")).strip()
    return ""


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
        source_role = _source_experiment_role(source_config_canonical)
        if source_role and source_role not in ALLOWED_SEED_SNAPSHOT_SOURCE_ROLES:
            raise RuntimeError(
                "Imported seed snapshot source must be a main or guided league-seed run, "
                f"got experiment.role={source_role!r}. "
                "Use --b1-baseline-run-dir for the strict B1 baseline instead."
            )
        source_config_sections = canonical_config_sections(source_config_canonical)
        expected_config_sections = canonical_config_sections(expected_config_canonical)
        for section_name in ("model", "environment"):
            source_section = source_config_sections.get(section_name)
            expected_section = expected_config_sections.get(section_name)
            if source_section is None or expected_section is None:
                continue
            if not config_sections_match_for_import_contract(
                section_name=section_name,
                source_section=source_section,
                expected_section=expected_section,
            ):
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

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported seed snapshot weights payload is missing model_state_dict: {source_run_dir}")
    missing, extra, _allowed_missing = state_dict_key_mismatch_for_context_compat(
        source_state_dict=source_model_state_dict,
        expected_state_dict=expected_model_state_dict,
    )
    if missing or extra:
        raise RuntimeError(
            "Imported seed snapshot model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(set(source_model_state_dict) & set(expected_model_state_dict)):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported seed snapshot tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def import_seed_snapshot_pool(
    *,
    stack: Any,
    training_paths: Any,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    source_run_dir = Path(seed_snapshot_run_dir).resolve()
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    source_registry_path = _seed_snapshot_registry_path(stack, source_run_dir=source_run_dir)
    if source_registry_path is None:
        source_registry_path = source_layout.training_snapshots_dir / REGISTRY_FILENAME
    if not source_registry_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve a snapshot registry in the seed snapshot run: {source_registry_path}"
        )
    source_registry = SnapshotRegistry.load(source_registry_path)
    source_champions = set(source_registry.champion_snapshots)
    source_pinned = set(source_registry.pinned_snapshots)
    import_filter = _seed_snapshot_import_filter(stack)
    source_snapshots = [
        snapshot
        for snapshot in source_registry.snapshots
        if snapshot.policy_id not in promotion_anchor_policy_id_candidates(PROMOTION_GATE_NOLEAGUE_BASELINE_NAME)
        and _should_import_seed_snapshot(
            policy=import_filter,
            source_policy_id=snapshot.policy_id,
            source_champions=source_champions,
            source_pinned=source_pinned,
        )
    ]
    if not source_snapshots:
        return []

    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    sync_snapshot_registry_retention(stack, registry)
    existing_policy_ids = {snapshot.policy_id for snapshot in registry.snapshots}
    champion_import_policy = _seed_snapshot_champion_import_policy(stack)
    imported_policy_ids: list[str] = []
    for source_snapshot in source_snapshots:
        imported_policy_id = _seed_snapshot_policy_id(
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
        )
        mark_imported_as_champion = _should_mark_imported_seed_snapshot_as_champion(
            policy=champion_import_policy,
            source_policy_id=source_snapshot.policy_id,
            source_champions=source_champions,
            source_pinned=source_pinned,
        )
        if imported_policy_id in existing_policy_ids:
            if mark_imported_as_champion and registry.has_snapshot(imported_policy_id):
                registry.add_champion(imported_policy_id)
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
        # Seed snapshots are external bootstrap material, not live-run progress.
        # Keep their source update as provenance while registering them at update
        # zero so live PFSP "recent" sampling quickly gives way to this run's own
        # snapshots instead of treating source-run update 75/100 as future-local
        # league progress.
        seed_registry_update = 0
        weights_path, weights_sha256 = write_imported_snapshot_artifact(
            snapshots_dir=training_paths.snapshots_dir,
            run_dir=run_dir,
            source_payload=payload,
            source_run_dir=source_run_dir,
            source_policy_id=source_snapshot.policy_id,
            source_snapshot_path=source_snapshot.path,
            target_policy_id=imported_policy_id,
            update=seed_registry_update,
            metadata_format="seeded_train_snapshot_metadata_v1",
            seeded_from_external_registry=True,
            imported_from_update=int(source_snapshot.update),
        )
        registry.add_snapshot(
            policy_id=imported_policy_id,
            update=seed_registry_update,
            weights_sha256=weights_sha256,
            path=weights_path.relative_to(run_dir).as_posix(),
        )
        if mark_imported_as_champion:
            registry.add_champion(imported_policy_id)
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
            f"source_run_dir={source_run_dir.as_posix()} "
            f"source_registry={source_registry_path.as_posix()}"
        )
    return imported_policy_ids


def _seed_snapshot_registry_path(stack: Any, *, source_run_dir: Path) -> Path | None:
    league = getattr(getattr(stack, "config", None), "league", None)
    pool = getattr(league, "pool", None)
    raw_path = str(getattr(pool, "seed_snapshot_registry_json", "") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate
    return (Path(source_run_dir) / path).resolve()


def _seed_snapshot_champion_import_policy(stack: Any) -> str:
    league = getattr(getattr(stack, "config", None), "league", None)
    pool = getattr(league, "pool", None)
    raw_policy = getattr(pool, "seed_snapshot_champion_import", "source_champions")
    policy = str(raw_policy).strip()
    if policy not in SEED_SNAPSHOT_CHAMPION_IMPORT_POLICIES:
        raise ValueError(
            "league.pool.seed_snapshot_champion_import must be one of: all, none, pinned, source_champions"
        )
    return policy


def _seed_snapshot_import_filter(stack: Any) -> str:
    league = getattr(getattr(stack, "config", None), "league", None)
    pool = getattr(league, "pool", None)
    raw_policy = getattr(pool, "seed_snapshot_import_filter", "all")
    policy = str(raw_policy).strip()
    if policy not in SEED_SNAPSHOT_IMPORT_FILTERS:
        raise ValueError(
            "league.pool.seed_snapshot_import_filter must be one of: "
            "all, none, pinned, pinned_or_source_champions, source_champions"
        )
    return policy


def _should_import_seed_snapshot(
    *,
    policy: str,
    source_policy_id: str,
    source_champions: set[str],
    source_pinned: set[str],
) -> bool:
    normalized = str(source_policy_id)
    if policy == "none":
        return False
    if policy == "all":
        return True
    if policy == "pinned":
        return normalized in source_pinned
    if policy == "source_champions":
        return normalized in source_champions
    return normalized in source_pinned or normalized in source_champions


def _should_mark_imported_seed_snapshot_as_champion(
    *,
    policy: str,
    source_policy_id: str,
    source_champions: set[str],
    source_pinned: set[str],
) -> bool:
    if policy == "none":
        return False
    if policy == "all":
        return True
    if policy == "pinned":
        return str(source_policy_id) in source_pinned
    return str(source_policy_id) in source_champions
