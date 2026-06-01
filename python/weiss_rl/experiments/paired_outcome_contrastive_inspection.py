"""Replay inspection for paired-outcome contrastive dataset generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_contrastive_sources import (
    PairedOutcomeContrastiveSource,
    PairedOutcomeInspectionSource,
)
from weiss_rl.replay.inspector import inspect_replay_bundle, write_replay_inspection_report
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset


@dataclass(frozen=True, slots=True)
class PairedOutcomeInspectionConfig:
    sources: tuple[PairedOutcomeInspectionSource, ...]
    stack_config: Path
    run_dir: Path
    policy_a: str
    policy_b: str
    snapshot_registry_json: Path | None = None
    top_k: int = 100_000
    top_actions: int = 3
    accepted_snapshot_config_hashes: tuple[str, ...] = ()
    max_bundles_per_source: int | None = None
    resume: bool = True


def inspect_paired_outcome_sources(
    config: PairedOutcomeInspectionConfig,
) -> tuple[tuple[PairedOutcomeContrastiveSource, ...], dict[str, Any]]:
    """Inspect policy A vs policy B on each source bundle and write JSON reports."""

    if not config.sources:
        raise ValueError("sources must contain at least one source")
    if int(config.top_k) < 0:
        raise ValueError("top_k must be >= 0")
    if int(config.top_actions) <= 0:
        raise ValueError("top_actions must be >= 1")
    if config.max_bundles_per_source is not None and int(config.max_bundles_per_source) <= 0:
        raise ValueError("max_bundles_per_source must be positive when provided")

    contrastive_sources: list[PairedOutcomeContrastiveSource] = []
    source_summaries: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for source in config.sources:
        dataset = load_replay_trajectory_bc_dataset(source.source_dataset_path)
        bundle_paths = _bundle_paths_from_dataset(dataset)
        if config.max_bundles_per_source is not None:
            bundle_paths = bundle_paths[: int(config.max_bundles_per_source)]
        if not bundle_paths:
            raise ValueError(f"source dataset has no selected bundle paths: {source.source_dataset_path}")

        inspection_dir = source.output_dir / "inspections"
        inspection_dir.mkdir(parents=True, exist_ok=True)
        inspection_jsons: list[Path] = []
        for bundle_path in bundle_paths:
            report_path = inspection_dir / f"{Path(bundle_path).stem}.json"
            if bool(config.resume) and report_path.is_file():
                counters["reused_inspections"] += 1
                inspection_jsons.append(report_path)
                continue
            report = inspect_replay_bundle(
                bundle_path=Path(bundle_path),
                stack=config.stack_config,
                policy_a=config.policy_a,
                policy_b=config.policy_b,
                run_dir=config.run_dir,
                snapshot_registry_path=config.snapshot_registry_json,
                top_k=int(config.top_k),
                top_actions=int(config.top_actions),
                accepted_snapshot_config_hashes=config.accepted_snapshot_config_hashes,
            )
            write_replay_inspection_report(report_path, report)
            counters["written_inspections"] += 1
            inspection_jsons.append(report_path)

        contrastive_sources.append(
            PairedOutcomeContrastiveSource(
                source_label=source.source_label,
                source_role=source.source_role,
                source_dataset_path=source.source_dataset_path,
                source_opponent_policy_id=source.source_opponent_policy_id,
                inspection_jsons=tuple(inspection_jsons),
            )
        )
        source_summaries.append(
            {
                "source_label": source.source_label,
                "source_role": source.source_role,
                "source_opponent_policy_id": source.source_opponent_policy_id,
                "source_dataset_path": source.source_dataset_path.as_posix(),
                "inspection_dir": inspection_dir.as_posix(),
                "inspection_count": len(inspection_jsons),
                "source_bundle_count": int(dataset.metadata.get("bundle_count", len(bundle_paths))),
                "inspected_bundle_count": len(bundle_paths),
            }
        )

    summary = {
        "kind": "paired_outcome_policy_inspections_v1",
        "stack_config": config.stack_config.as_posix(),
        "run_dir": config.run_dir.as_posix(),
        "snapshot_registry_json": None
        if config.snapshot_registry_json is None
        else config.snapshot_registry_json.as_posix(),
        "policy_a": str(config.policy_a),
        "policy_b": str(config.policy_b),
        "top_k": int(config.top_k),
        "top_actions": int(config.top_actions),
        "accepted_snapshot_config_hashes": list(config.accepted_snapshot_config_hashes),
        "max_bundles_per_source": config.max_bundles_per_source,
        "source_count": len(config.sources),
        "sources": source_summaries,
        "counters": dict(sorted(counters.items())),
    }
    return tuple(contrastive_sources), summary


def _bundle_paths_from_dataset(dataset: ReplayTrajectoryDataset) -> tuple[Path, ...]:
    paths: list[Path] = []
    raw_bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(raw_bundles, list):
        return ()
    for bundle in raw_bundles:
        if not isinstance(bundle, Mapping):
            continue
        raw_path = bundle.get("bundle_path")
        if isinstance(raw_path, str) and raw_path.strip():
            paths.append(Path(raw_path))
    return tuple(paths)


__all__ = [
    "PairedOutcomeInspectionConfig",
    "inspect_paired_outcome_sources",
]
