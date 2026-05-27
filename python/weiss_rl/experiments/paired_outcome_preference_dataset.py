"""Build explicit preferred/rejected trajectory preference replay datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    merge_replay_trajectory_bc_datasets,
    save_replay_trajectory_bc_dataset,
    subset_replay_trajectory_bc_dataset,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceDatasetConfig:
    preferred_dataset: Path
    rejected_dataset: Path
    output_dataset: Path
    output_summary_json: Path | None = None
    max_pairs: int | None = None
    preferred_label: str = "preferred"
    rejected_label: str = "rejected"
    opponent_match_aliases: Mapping[str, str] | None = None


def build_paired_outcome_preference_dataset(
    config: PairedOutcomePreferenceDatasetConfig,
) -> tuple[ReplayTrajectoryDataset, dict[str, Any]]:
    preferred = load_replay_trajectory_bc_dataset(config.preferred_dataset)
    rejected = load_replay_trajectory_bc_dataset(config.rejected_dataset)
    aliases = _normalize_opponent_match_aliases(config.opponent_match_aliases)
    pairs = _matched_episode_pairs(
        preferred=preferred,
        rejected=rejected,
        max_pairs=config.max_pairs,
        opponent_match_aliases=aliases,
    )
    if not pairs:
        raise ValueError("no matching preferred/rejected episode pairs found")

    preferred_indices = [pair["preferred_episode_index"] for pair in pairs]
    rejected_indices = [pair["rejected_episode_index"] for pair in pairs]
    preferred_bundles = _annotated_bundles(
        preferred,
        episode_indices=preferred_indices,
        pairs=pairs,
        role=1,
        label=config.preferred_label,
    )
    rejected_bundles = _annotated_bundles(
        rejected,
        episode_indices=rejected_indices,
        pairs=pairs,
        role=0,
        label=config.rejected_label,
    )
    preferred_subset = subset_replay_trajectory_bc_dataset(
        preferred,
        episode_indices=preferred_indices,
        selected_bundles=preferred_bundles,
        metadata_updates={"preference_role": "preferred"},
    )
    rejected_subset = subset_replay_trajectory_bc_dataset(
        rejected,
        episode_indices=rejected_indices,
        selected_bundles=rejected_bundles,
        metadata_updates={"preference_role": "rejected"},
    )
    merged = merge_replay_trajectory_bc_datasets(
        [preferred_subset, rejected_subset],
        source_labels=(config.preferred_label, config.rejected_label),
        preserve_source_bundle_labels=True,
        offset_preference_pair_ids=False,
    )
    summary = {
        "kind": "paired_outcome_preference_dataset_v1",
        "preferred_dataset": config.preferred_dataset.as_posix(),
        "rejected_dataset": config.rejected_dataset.as_posix(),
        "output_dataset": config.output_dataset.as_posix(),
        "preferred_label": config.preferred_label,
        "rejected_label": config.rejected_label,
        "opponent_match_aliases": dict(sorted(aliases.items())),
        "pair_count": len(pairs),
        "pairs": pairs,
        "preferred_train_rows": int(preferred_subset.metadata.get("train_rows", 0)),
        "rejected_train_rows": int(rejected_subset.metadata.get("train_rows", 0)),
        "merged_train_rows": int(merged.metadata.get("train_rows", 0)),
    }
    merged.metadata["paired_outcome_preference_generation"] = summary
    save_replay_trajectory_bc_dataset(config.output_dataset, merged)
    if config.output_summary_json is not None:
        config.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged, summary


def _matched_episode_pairs(
    *,
    preferred: ReplayTrajectoryDataset,
    rejected: ReplayTrajectoryDataset,
    max_pairs: int | None,
    opponent_match_aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    preferred_by_key = _episode_indices_by_key(preferred, opponent_match_aliases=opponent_match_aliases)
    rejected_by_key = _episode_indices_by_key(rejected, opponent_match_aliases=opponent_match_aliases)
    pairs: list[dict[str, Any]] = []
    for key in sorted(set(preferred_by_key) & set(rejected_by_key)):
        preferred_indices = preferred_by_key[key]
        rejected_indices = rejected_by_key[key]
        count = min(len(preferred_indices), len(rejected_indices))
        for offset in range(count):
            preference_pair_id = len(pairs)
            pairs.append(
                {
                    "preference_pair_id": preference_pair_id,
                    "match_key": list(key),
                    "preferred_episode_index": int(preferred_indices[offset]),
                    "rejected_episode_index": int(rejected_indices[offset]),
                }
            )
            if max_pairs is not None and len(pairs) >= int(max_pairs):
                return pairs
    return pairs


def _episode_indices_by_key(
    dataset: ReplayTrajectoryDataset,
    *,
    opponent_match_aliases: Mapping[str, str],
) -> dict[tuple[str, str, str, str], list[int]]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return {}
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping):
            continue
        key = _preference_match_key(bundle, opponent_match_aliases=opponent_match_aliases)
        if key is None:
            continue
        grouped[key].append(int(index))
    return {key: values for key, values in grouped.items() if values}


def _preference_match_key(
    bundle: Mapping[str, Any],
    *,
    opponent_match_aliases: Mapping[str, str],
) -> tuple[str, str, str, str] | None:
    opponent = str(bundle.get("source_opponent_policy_id") or bundle.get("opponent_policy_id") or "").strip()
    pair_index = _first_present_text(bundle, ("source_pair_index", "pair_index"))
    swap_index = _first_present_text(bundle, ("source_swap_index", "swap_index"))
    episode_seed = _first_present_text(bundle, ("episode_seed", "source_episode_seed"))
    if not opponent or (pair_index is None and episode_seed is None):
        return None
    opponent = _canonical_opponent_match_id(opponent, opponent_match_aliases=opponent_match_aliases)
    return (
        opponent,
        "" if pair_index is None else pair_index,
        "" if swap_index is None else swap_index,
        "" if episode_seed is None else episode_seed,
    )


def _first_present_text(bundle: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = bundle.get(key)
        if value is not None:
            return str(value)
    return None


def _annotated_bundles(
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: Sequence[int],
    pairs: Sequence[Mapping[str, Any]],
    role: int,
    label: str,
) -> list[dict[str, Any]]:
    raw_bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(raw_bundles, list):
        raise ValueError("preference dataset source is missing selected_bundles metadata")
    result: list[dict[str, Any]] = []
    for output_index, episode_index in enumerate(episode_indices):
        raw_bundle = raw_bundles[int(episode_index)]
        bundle = dict(raw_bundle) if isinstance(raw_bundle, Mapping) else {}
        pair = pairs[output_index]
        bundle["preference_pair_id"] = int(pair["preference_pair_id"])
        bundle["preference_role"] = int(role)
        bundle["preference_role_label"] = label
        bundle["preference_match_key"] = list(pair["match_key"])
        bundle["preference_match_opponent_policy_id"] = str(pair["match_key"][0])
        result.append(bundle)
    return result


def _normalize_opponent_match_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for source, target in (aliases or {}).items():
        source_text = str(source).strip()
        target_text = str(target).strip()
        if not source_text or not target_text:
            continue
        normalized[source_text] = target_text
    return normalized


def _canonical_opponent_match_id(
    opponent_policy_id: str,
    *,
    opponent_match_aliases: Mapping[str, str],
) -> str:
    current = str(opponent_policy_id).strip()
    seen: set[str] = set()
    while current in opponent_match_aliases and current not in seen:
        seen.add(current)
        current = str(opponent_match_aliases[current]).strip()
    return current


__all__ = [
    "PairedOutcomePreferenceDatasetConfig",
    "build_paired_outcome_preference_dataset",
]
