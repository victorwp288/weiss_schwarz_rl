"""Build replay-BC data from exact paired outcome flip reports."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.experiments.champion_hardneg_trajectory_bc import (
    build_champion_hardneg_trajectory_bc_dataset,
    normalize_explicit_paired_seeds,
    normalize_include_outcomes,
    slug_policy_id,
)
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    merge_replay_trajectory_bc_datasets,
    save_replay_trajectory_bc_dataset,
)

_SCRIPT_KIND = "paired_flip_trajectory_bc_dataset_v1"


@dataclass(frozen=True, slots=True)
class PairedFlipTrajectoryBcConfig:
    stack: StackConfig
    contract: SimulatorContract
    stack_config: Path
    run_dir: Path
    snapshot_registry_json: Path
    paired_flip_targets_json: Path
    focal_policy_id: str
    output_run_dir: Path
    output_dataset: Path
    include_outcomes: tuple[str, ...] = ("W",)
    b1_baseline_run_dir: Path | None = None
    hard_negative_policy_ids: tuple[str, ...] = ()
    source_label_prefix: str = ""


def paired_flip_opponent_seed_plan(report: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return exact opponent -> paired seed mapping from a paired-flip report."""

    targets = report.get("targets")
    if not isinstance(targets, list):
        raise ValueError("paired flip report missing targets list")
    grouped: dict[str, set[int]] = defaultdict(set)
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        opponent = str(target.get("opponent_policy_id", "")).strip()
        raw_seed = target.get("episode_seed")
        if not opponent or raw_seed is None:
            continue
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"paired flip target has invalid episode_seed: {raw_seed!r}") from exc
        if seed < 0:
            raise ValueError(f"paired flip target has negative episode_seed: {seed}")
        grouped[opponent].add(seed)
    return {opponent: tuple(sorted(seeds)) for opponent, seeds in sorted(grouped.items()) if seeds}


def paired_flip_target_metadata_by_opponent_seed(
    report: Mapping[str, Any],
) -> dict[str, dict[int, tuple[dict[str, Any], ...]]]:
    """Return exact target provenance keyed by opponent and episode seed."""

    targets = report.get("targets")
    if not isinstance(targets, list):
        raise ValueError("paired flip report missing targets list")
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        selection = {}
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        opponent = str(target.get("opponent_policy_id", "")).strip()
        raw_seed = target.get("episode_seed")
        if not opponent or raw_seed is None:
            continue
        try:
            seed = int(raw_seed)
            pair_index = int(target["pair_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"paired flip target has invalid pair/seed provenance: {target!r}") from exc
        if seed < 0:
            raise ValueError(f"paired flip target has negative episode_seed: {seed}")
        swap_index = target.get("swap_index")
        try:
            normalized_swap = None if swap_index is None else int(swap_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"paired flip target has invalid swap_index: {swap_index!r}") from exc
        grouped[opponent][seed].append(
            {
                "target_id": str(target.get("target_id") or ""),
                "pair_index": pair_index,
                "pair_index_bucket": _source_pair_index_bucket(pair_index, selection=selection),
                "swap_index": normalized_swap,
                "episode_seed": seed,
                "flip_kind": selection.get("flip_kind"),
                "tags": list(target.get("tags") or ()),
            }
        )
    return {
        opponent: {seed: tuple(items) for seed, items in sorted(seeds.items())}
        for opponent, seeds in sorted(grouped.items())
        if seeds
    }


def build_paired_flip_trajectory_bc_dataset(
    config: PairedFlipTrajectoryBcConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    """Rerun exact paired-flip seeds and merge winner trajectory datasets."""

    report = _read_json_object(config.paired_flip_targets_json)
    seed_plan = paired_flip_opponent_seed_plan(report)
    if not seed_plan:
        raise ValueError(f"paired flip report contains no usable targets: {config.paired_flip_targets_json}")
    target_metadata_by_seed = paired_flip_target_metadata_by_opponent_seed(report)

    include_outcomes = normalize_include_outcomes(config.include_outcomes)
    output_run_dir = Path(config.output_run_dir)
    sources_dir = output_run_dir / "paired_flip_sources"
    datasets_dir = output_run_dir / "paired_flip_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[ReplayTrajectoryDataset] = []
    source_labels: list[str] = []
    source_summaries: list[dict[str, Any]] = []
    for opponent, seeds in seed_plan.items():
        normalized_seeds = normalize_explicit_paired_seeds(seeds)
        slug = slug_policy_id(opponent)
        source_run_dir = sources_dir / slug
        source_dataset_path = datasets_dir / f"paired_flip_bc_{slug}.npz"
        dataset, source_summary = build_champion_hardneg_trajectory_bc_dataset(
            stack=config.stack,
            contract=config.contract,
            stack_config=config.stack_config,
            run_dir=config.run_dir,
            output_run_dir=source_run_dir,
            output_dataset=source_dataset_path,
            snapshot_registry_json=config.snapshot_registry_json,
            focal_policy_id=str(config.focal_policy_id),
            opponent_policy_ids=(opponent,),
            paired_seed_count=len(normalized_seeds),
            include_outcomes=include_outcomes,
            b1_baseline_run_dir=config.b1_baseline_run_dir,
            hard_negative_policy_ids=config.hard_negative_policy_ids,
            explicit_paired_seeds=normalized_seeds,
        )
        _annotate_source_pair_metadata(
            dataset,
            opponent=opponent,
            targets_by_seed=target_metadata_by_seed.get(opponent, {}),
            paired_flip_targets_json=config.paired_flip_targets_json,
        )
        label_prefix = str(config.source_label_prefix).strip()
        source_label = f"{label_prefix}{opponent}" if label_prefix else opponent
        dataset.metadata["paired_flip_source_label"] = source_label
        dataset.metadata["paired_flip_target_report"] = config.paired_flip_targets_json.as_posix()
        dataset.metadata["paired_flip_target_seeds"] = [int(seed) for seed in normalized_seeds]
        save_replay_trajectory_bc_dataset(source_dataset_path, dataset)
        datasets.append(dataset)
        source_labels.append(source_label)
        source_summaries.append(
            {
                "opponent_policy_id": opponent,
                "source_label": source_label,
                "paired_seeds": [int(seed) for seed in normalized_seeds],
                "dataset_path": source_dataset_path.as_posix(),
                "output_run_dir": source_run_dir.as_posix(),
                "summary": source_summary,
                "train_rows": int(dataset.metadata.get("train_rows", 0)),
                "bundle_count": int(dataset.metadata.get("bundle_count", 0)),
            }
        )

    merged = (
        datasets[0]
        if len(datasets) == 1
        else merge_replay_trajectory_bc_datasets(datasets, source_labels=source_labels)
    )
    generation = {
        "kind": _SCRIPT_KIND,
        "paired_flip_targets_json": config.paired_flip_targets_json.as_posix(),
        "source_report_kind": report.get("kind"),
        "source_flip_kind": (report.get("selection") or {}).get("flip_kind")
        if isinstance(report.get("selection"), Mapping)
        else None,
        "focal_policy_id": str(config.focal_policy_id),
        "stack_config": config.stack_config.as_posix(),
        "run_dir": config.run_dir.as_posix(),
        "snapshot_registry_json": config.snapshot_registry_json.as_posix(),
        "b1_baseline_run_dir": None if config.b1_baseline_run_dir is None else config.b1_baseline_run_dir.as_posix(),
        "include_outcomes": list(include_outcomes),
        "opponent_seed_plan": {opponent: [int(seed) for seed in seeds] for opponent, seeds in seed_plan.items()},
        "sources": source_summaries,
    }
    merged.metadata["paired_flip_generation"] = generation
    config.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    save_replay_trajectory_bc_dataset(config.output_dataset, merged)
    return merged, {"generation": generation, "dataset_path": config.output_dataset.as_posix()}


def write_paired_flip_trajectory_bc_summary(
    path: Path,
    *,
    summary: Mapping[str, Any],
    dataset: ReplayTrajectoryDataset,
) -> None:
    payload = {
        **dict(summary),
        "dataset_metadata": dict(dataset.metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _annotate_source_pair_metadata(
    dataset: ReplayTrajectoryDataset,
    *,
    opponent: str,
    targets_by_seed: Mapping[int, tuple[Mapping[str, Any], ...]],
    paired_flip_targets_json: Path,
) -> None:
    raw_bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(raw_bundles, list):
        return
    for bundle in raw_bundles:
        if not isinstance(bundle, dict):
            continue
        raw_seed = bundle.get("episode_seed")
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            continue
        target_items = tuple(targets_by_seed.get(seed, ()))
        if not target_items:
            continue
        pair_indices = sorted({int(item["pair_index"]) for item in target_items})
        buckets = tuple(dict.fromkeys(str(item.get("pair_index_bucket") or "") for item in target_items))
        target_ids = tuple(
            str(item.get("target_id") or "") for item in target_items if str(item.get("target_id") or "")
        )
        swap_indices = tuple(int(item["swap_index"]) for item in target_items if item.get("swap_index") is not None)
        bundle["paired_compare_source_json"] = Path(paired_flip_targets_json).as_posix()
        bundle["source_opponent_policy_id"] = str(opponent)
        bundle["source_pair_indices"] = pair_indices
        if len(pair_indices) == 1:
            bundle["source_pair_index"] = int(pair_indices[0])
        bundle["source_pair_index_buckets"] = list(buckets)
        if len(buckets) == 1:
            bundle["source_pair_index_bucket"] = buckets[0]
        bundle["source_target_ids"] = list(target_ids)
        bundle["source_swap_indices"] = list(swap_indices)
        bundle["source_target_count"] = len(target_items)
        flip_kinds = tuple(dict.fromkeys(str(item.get("flip_kind") or "") for item in target_items))
        if len(flip_kinds) == 1:
            bundle["source_flip_kind"] = flip_kinds[0]


def _source_pair_index_bucket(pair_index: int, *, selection: Mapping[str, Any]) -> str:
    raw_min = selection.get("pair_index_min")
    raw_max = selection.get("pair_index_max")
    pair_min = None if raw_min is None else int(raw_min)
    pair_max = None if raw_max is None else int(raw_max)
    if pair_min is not None and pair_max is None:
        return f"pair_index_gte_{pair_min}" if int(pair_index) >= pair_min else f"pair_index_lt_{pair_min}"
    if pair_min is None and pair_max is not None:
        return f"pair_index_lte_{pair_max}" if int(pair_index) <= pair_max else f"pair_index_gt_{pair_max}"
    if pair_min is not None and pair_max is not None:
        if pair_min <= int(pair_index) <= pair_max:
            return f"pair_index_{pair_min}_to_{pair_max}"
        if int(pair_index) < pair_min:
            return f"pair_index_lt_{pair_min}"
        return f"pair_index_gt_{pair_max}"
    return "pair_index_all"


__all__ = [
    "PairedFlipTrajectoryBcConfig",
    "build_paired_flip_trajectory_bc_dataset",
    "paired_flip_opponent_seed_plan",
    "paired_flip_target_metadata_by_opponent_seed",
    "write_paired_flip_trajectory_bc_summary",
]
