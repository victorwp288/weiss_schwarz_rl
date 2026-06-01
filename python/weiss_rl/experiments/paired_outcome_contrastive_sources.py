"""Source discovery for paired-outcome contrastive replay datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.champion_hardneg_trajectory_bc import slug_policy_id


@dataclass(frozen=True, slots=True)
class PairedOutcomeContrastiveSource:
    source_label: str
    source_role: str
    source_dataset_path: Path
    inspection_jsons: tuple[Path, ...]
    source_opponent_policy_id: str = ""


@dataclass(frozen=True, slots=True)
class PairedOutcomeInspectionSource:
    source_label: str
    source_role: str
    source_dataset_path: Path
    source_opponent_policy_id: str
    output_dir: Path


def sources_from_paired_flip_summary(
    summary_json: Path,
    *,
    source_role: str,
    output_dir: Path,
    include_source_labels: Sequence[str] = (),
) -> tuple[PairedOutcomeInspectionSource, ...]:
    """Return source records from a paired-flip trajectory-BC summary."""

    payload = _read_json_object(summary_json)
    generation = payload.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError(f"paired-flip summary is missing generation object: {summary_json}")
    raw_sources = generation.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError(f"paired-flip summary is missing generation.sources: {summary_json}")
    include = {str(item).strip() for item in include_source_labels if str(item).strip()}
    sources: list[PairedOutcomeInspectionSource] = []
    for item in raw_sources:
        if not isinstance(item, Mapping):
            continue
        source_label = str(item.get("source_label") or item.get("opponent_policy_id") or "").strip()
        opponent = str(item.get("opponent_policy_id") or "").strip()
        if not source_label:
            raise ValueError(f"paired-flip source is missing source_label/opponent_policy_id: {summary_json}")
        if include and source_label not in include and opponent not in include:
            continue
        raw_dataset_path = item.get("dataset_path")
        if not isinstance(raw_dataset_path, str) or not raw_dataset_path.strip():
            raise ValueError(f"paired-flip source {source_label!r} is missing dataset_path")
        dataset_path = _resolve_paired_flip_source_dataset_path(Path(raw_dataset_path), source=item)
        sources.append(
            PairedOutcomeInspectionSource(
                source_label=source_label,
                source_role=str(source_role).strip() or "unspecified",
                source_dataset_path=dataset_path,
                source_opponent_policy_id=opponent,
                output_dir=Path(output_dir) / slug_policy_id(source_label),
            )
        )
    if not sources:
        raise ValueError(f"no paired-flip sources selected from {summary_json}")
    return tuple(sources)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve_paired_flip_source_dataset_path(path: Path, *, source: Mapping[str, Any]) -> Path:
    if path.is_file():
        return path
    raw_output_run_dir = source.get("output_run_dir")
    if isinstance(raw_output_run_dir, str) and raw_output_run_dir.strip():
        datasets_dir = Path(raw_output_run_dir) / "datasets"
        if datasets_dir.is_dir():
            candidates = sorted(datasets_dir.glob("*.npz"))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                slug = slug_policy_id(str(source.get("opponent_policy_id") or source.get("source_label") or ""))
                slug_matches = [candidate for candidate in candidates if slug in candidate.stem]
                if len(slug_matches) == 1:
                    return slug_matches[0]
    return path


__all__ = [
    "PairedOutcomeContrastiveSource",
    "PairedOutcomeInspectionSource",
    "sources_from_paired_flip_summary",
]
