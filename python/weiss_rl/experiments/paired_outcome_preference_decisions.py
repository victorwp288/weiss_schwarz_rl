"""Decision-level diagnostics for paired outcome preference replay datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset
from weiss_rl.training.run_metadata import load_json_object


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceDecisionConfig:
    dataset_path: Path
    spec_bundle_json: Path | None = None
    max_examples: int = 25
    top_action_edges: int = 25


def build_paired_outcome_preference_decision_report(
    config: PairedOutcomePreferenceDecisionConfig,
) -> dict[str, Any]:
    """Return pair/action/conflict diagnostics for a paired preference replay dataset."""

    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    bundles = _selected_bundles(dataset)
    catalog = _load_optional_action_catalog(config.spec_bundle_json)
    rows = _iter_train_rows(dataset, bundles=bundles, catalog=catalog)
    pair_summaries, aligned_edges = _pair_summaries(
        dataset,
        bundles=bundles,
        catalog=catalog,
        max_examples=int(config.max_examples),
    )
    current_state_edges = [edge for edge in aligned_edges if bool(edge["same_current_state"])]
    history_edges = [edge for edge in aligned_edges if bool(edge["same_history"])]
    current_state_diff_edges = [edge for edge in current_state_edges if not bool(edge["same_action"])]
    history_diff_edges = [edge for edge in history_edges if not bool(edge["same_action"])]
    return {
        "kind": "paired_outcome_preference_decision_report_v1",
        "dataset_path": config.dataset_path.as_posix(),
        "spec_bundle_json": None if config.spec_bundle_json is None else config.spec_bundle_json.as_posix(),
        "episode_count": int(dataset.episode_count),
        "train_rows": int(np.count_nonzero(dataset.policy_train_mask)),
        "opponent_policy_episode_count": _nonempty_episode_metadata_count(bundles, "source_opponent_policy_id"),
        "preference_pair_count": len(pair_summaries),
        "complete_pair_count": sum(1 for row in pair_summaries if bool(row["complete_pair"])),
        "incomplete_pair_count": sum(1 for row in pair_summaries if not bool(row["complete_pair"])),
        "aligned_step_count": len(aligned_edges),
        "aligned_different_action_count": sum(1 for edge in aligned_edges if not bool(edge["same_action"])),
        "same_current_state_edge_count": len(current_state_edges),
        "same_history_edge_count": len(history_edges),
        "same_current_state_different_action_edge_count": len(current_state_diff_edges),
        "same_history_different_action_edge_count": len(history_diff_edges),
        "current_state_conflict_count": len(
            _conflicts_for_edges(
                current_state_diff_edges, key_name="current_state_hash", max_examples=int(config.max_examples)
            )
        ),
        "history_conflict_count": len(
            _conflicts_for_edges(history_diff_edges, key_name="history_hash", max_examples=int(config.max_examples))
        ),
        "role_summaries": _role_summaries(rows),
        "opponent_summaries": _opponent_summaries(rows),
        "action_edge_counts": _action_edge_counts(
            aligned_edges,
            catalog=catalog,
            limit=int(config.top_action_edges),
            same_current_state_only=False,
        ),
        "same_current_state_action_edge_counts": _action_edge_counts(
            current_state_edges,
            catalog=catalog,
            limit=int(config.top_action_edges),
            same_current_state_only=True,
        ),
        "current_state_conflicts": _conflicts_for_edges(
            current_state_diff_edges,
            key_name="current_state_hash",
            max_examples=int(config.max_examples),
        ),
        "history_conflicts": _conflicts_for_edges(
            history_diff_edges,
            key_name="history_hash",
            max_examples=int(config.max_examples),
        ),
        "pair_summaries": pair_summaries,
    }


def write_paired_outcome_preference_decision_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_bundles(dataset: ReplayTrajectoryDataset) -> list[Mapping[str, Any]]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        raise ValueError("dataset metadata must contain selected_bundles matching episode_count")
    result: list[Mapping[str, Any]] = []
    for bundle in bundles:
        result.append(bundle if isinstance(bundle, Mapping) else {})
    return result


def _iter_train_rows(
    dataset: ReplayTrajectoryDataset,
    *,
    bundles: Sequence[Mapping[str, Any]],
    catalog: ActionCatalog | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_index, episode_index in zip(*np.nonzero(dataset.policy_train_mask.astype(bool)), strict=False):
        step = int(step_index)
        episode = int(episode_index)
        bundle = bundles[episode]
        action = int(dataset.actions[step, episode])
        rows.append(
            {
                "step_index": step,
                "episode_index": episode,
                "preference_pair_id": _optional_int(bundle.get("preference_pair_id")),
                "preference_role": _optional_int(bundle.get("preference_role")),
                "preference_role_label": str(bundle.get("preference_role_label") or ""),
                "source_opponent_policy_id": str(bundle.get("source_opponent_policy_id") or ""),
                "source_pair_index": _first_optional_int(bundle, ("source_pair_index", "pair_index")),
                "episode_seed": _first_optional_int(bundle, ("episode_seed", "source_episode_seed")),
                "swap_index": _first_optional_int(bundle, ("source_swap_index", "swap_index")),
                "focal_seat": _optional_int(bundle.get("focal_seat")),
                "action": action,
                "decoded_action": _decoded_action(action, catalog),
                "current_state_hash": _state_hash(dataset, step_index=step, episode_index=episode),
                "history_hash": _history_hash(dataset, step_index=step, episode_index=episode),
            }
        )
    return rows


def _pair_summaries(
    dataset: ReplayTrajectoryDataset,
    *,
    bundles: Sequence[Mapping[str, Any]],
    catalog: ActionCatalog | None,
    max_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes_by_pair_role: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for episode_index, bundle in enumerate(bundles):
        pair_id = _optional_int(bundle.get("preference_pair_id"))
        role = _optional_int(bundle.get("preference_role"))
        if pair_id is None or role not in {0, 1}:
            continue
        episodes_by_pair_role[int(pair_id)][int(role)].append(int(episode_index))

    summaries: list[dict[str, Any]] = []
    aligned_edges: list[dict[str, Any]] = []
    for pair_id in sorted(episodes_by_pair_role):
        by_role = episodes_by_pair_role[pair_id]
        preferred = list(by_role.get(1, []))
        rejected = list(by_role.get(0, []))
        complete = bool(preferred and rejected)
        preferred_rows = _train_row_count(dataset, preferred)
        rejected_rows = _train_row_count(dataset, rejected)
        pair_edges: list[dict[str, Any]] = []
        if complete:
            pair_edges = _aligned_edges_for_pair(
                dataset,
                bundles=bundles,
                catalog=catalog,
                preference_pair_id=pair_id,
                preferred_episode=int(preferred[0]),
                rejected_episode=int(rejected[0]),
            )
            aligned_edges.extend(pair_edges)
        diff_edges = [edge for edge in pair_edges if not bool(edge["same_action"])]
        same_state_edges = [edge for edge in diff_edges if bool(edge["same_current_state"])]
        same_history_edges = [edge for edge in diff_edges if bool(edge["same_history"])]
        bundle = bundles[preferred[0]] if preferred else (bundles[rejected[0]] if rejected else {})
        summaries.append(
            {
                "preference_pair_id": int(pair_id),
                "complete_pair": complete,
                "preferred_episode_indices": preferred,
                "rejected_episode_indices": rejected,
                "preferred_train_rows": preferred_rows,
                "rejected_train_rows": rejected_rows,
                "aligned_step_count": len(pair_edges),
                "aligned_different_action_count": len(diff_edges),
                "same_current_state_different_action_count": len(same_state_edges),
                "same_history_different_action_count": len(same_history_edges),
                "source_opponent_policy_id": str(bundle.get("source_opponent_policy_id") or ""),
                "source_pair_index": _first_optional_int(bundle, ("source_pair_index", "pair_index")),
                "episode_seed": _first_optional_int(bundle, ("episode_seed", "source_episode_seed")),
                "top_action_edges": _action_edge_counts(
                    diff_edges,
                    catalog=catalog,
                    limit=max(0, max_examples),
                    same_current_state_only=False,
                ),
                "examples": diff_edges[: max(0, max_examples)],
            }
        )
    return summaries, aligned_edges


def _aligned_edges_for_pair(
    dataset: ReplayTrajectoryDataset,
    *,
    bundles: Sequence[Mapping[str, Any]],
    catalog: ActionCatalog | None,
    preference_pair_id: int,
    preferred_episode: int,
    rejected_episode: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    max_step = int(dataset.time_steps)
    preferred_bundle = bundles[preferred_episode]
    rejected_bundle = bundles[rejected_episode]
    for step in range(max_step):
        if not bool(dataset.policy_train_mask[step, preferred_episode]):
            continue
        if not bool(dataset.policy_train_mask[step, rejected_episode]):
            continue
        preferred_action = int(dataset.actions[step, preferred_episode])
        rejected_action = int(dataset.actions[step, rejected_episode])
        preferred_state = _state_hash(dataset, step_index=step, episode_index=preferred_episode)
        rejected_state = _state_hash(dataset, step_index=step, episode_index=rejected_episode)
        preferred_history = _history_hash(dataset, step_index=step, episode_index=preferred_episode)
        rejected_history = _history_hash(dataset, step_index=step, episode_index=rejected_episode)
        edges.append(
            {
                "preference_pair_id": int(preference_pair_id),
                "step_index": int(step),
                "preferred_episode_index": int(preferred_episode),
                "rejected_episode_index": int(rejected_episode),
                "preferred_action": preferred_action,
                "rejected_action": rejected_action,
                "preferred_decoded_action": _decoded_action(preferred_action, catalog),
                "rejected_decoded_action": _decoded_action(rejected_action, catalog),
                "same_action": preferred_action == rejected_action,
                "same_family": _action_family(preferred_action, catalog) == _action_family(rejected_action, catalog),
                "same_current_state": preferred_state == rejected_state,
                "same_history": preferred_history == rejected_history,
                "current_state_hash": preferred_state if preferred_state == rejected_state else None,
                "preferred_state_hash": preferred_state,
                "rejected_state_hash": rejected_state,
                "history_hash": preferred_history if preferred_history == rejected_history else None,
                "preferred_history_hash": preferred_history,
                "rejected_history_hash": rejected_history,
                "source_opponent_policy_id": str(preferred_bundle.get("source_opponent_policy_id") or ""),
                "rejected_source_opponent_policy_id": str(rejected_bundle.get("source_opponent_policy_id") or ""),
                "source_pair_index": _first_optional_int(preferred_bundle, ("source_pair_index", "pair_index")),
                "episode_seed": _first_optional_int(preferred_bundle, ("episode_seed", "source_episode_seed")),
                "swap_index": _first_optional_int(preferred_bundle, ("source_swap_index", "swap_index")),
                "preferred_role_label": str(preferred_bundle.get("preference_role_label") or ""),
                "rejected_role_label": str(rejected_bundle.get("preference_role_label") or ""),
            }
        )
    return edges


def _action_edge_counts(
    edges: Sequence[Mapping[str, Any]],
    *,
    catalog: ActionCatalog | None,
    limit: int,
    same_current_state_only: bool,
) -> list[dict[str, Any]]:
    counter: Counter[tuple[int, int]] = Counter()
    same_state_counter: Counter[tuple[int, int]] = Counter()
    same_history_counter: Counter[tuple[int, int]] = Counter()
    for edge in edges:
        if bool(edge.get("same_action")):
            continue
        if same_current_state_only and not bool(edge.get("same_current_state")):
            continue
        key = (int(edge["preferred_action"]), int(edge["rejected_action"]))
        counter[key] += 1
        if bool(edge.get("same_current_state")):
            same_state_counter[key] += 1
        if bool(edge.get("same_history")):
            same_history_counter[key] += 1
    rows: list[dict[str, Any]] = []
    for (preferred_action, rejected_action), count in counter.most_common(max(0, limit)):
        rows.append(
            {
                "preferred_action": preferred_action,
                "rejected_action": rejected_action,
                "count": int(count),
                "same_current_state_count": int(same_state_counter[(preferred_action, rejected_action)]),
                "same_history_count": int(same_history_counter[(preferred_action, rejected_action)]),
                "preferred": _decoded_action(preferred_action, catalog),
                "rejected": _decoded_action(rejected_action, catalog),
            }
        )
    return rows


def _conflicts_for_edges(
    edges: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        key = str(edge.get(key_name) or "")
        if key:
            grouped[key].append(edge)

    conflicts: list[dict[str, Any]] = []
    for key, group_edges in sorted(grouped.items()):
        preferred_actions = sorted({int(edge["preferred_action"]) for edge in group_edges})
        rejected_actions = sorted({int(edge["rejected_action"]) for edge in group_edges})
        exact_reverse_pairs = 0
        for left_index, left in enumerate(group_edges):
            for right in group_edges[left_index + 1 :]:
                if int(left["preferred_action"]) == int(right["rejected_action"]) and int(
                    left["rejected_action"]
                ) == int(right["preferred_action"]):
                    exact_reverse_pairs += 1
        if len(preferred_actions) <= 1 and exact_reverse_pairs <= 0:
            continue
        conflicts.append(
            {
                key_name: key,
                "edge_count": len(group_edges),
                "preferred_actions": preferred_actions,
                "rejected_actions": rejected_actions,
                "exact_reverse_pair_count": exact_reverse_pairs,
                "opponent_policy_ids": sorted(
                    {str(edge.get("source_opponent_policy_id") or "") for edge in group_edges}
                ),
                "source_pair_indices": sorted(
                    {
                        int(edge["source_pair_index"])
                        for edge in group_edges
                        if edge.get("source_pair_index") is not None
                    }
                ),
                "examples": [dict(edge) for edge in group_edges[: max(0, max_examples)]],
            }
        )
    return conflicts


def _role_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("preference_role_label") or row.get("preference_role") or "")].append(row)
    return [
        {
            "label": label,
            "row_count": len(group),
            "action_counts": _raw_action_counts(group),
        }
        for label, group in sorted(grouped.items())
    ]


def _opponent_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_opponent_policy_id") or "")].append(row)
    return [
        {
            "opponent_policy_id": label,
            "row_count": len(group),
            "pair_indices": sorted(
                {int(row["source_pair_index"]) for row in group if row.get("source_pair_index") is not None}
            ),
            "action_counts": _raw_action_counts(group),
        }
        for label, group in sorted(grouped.items())
    ]


def _raw_action_counts(rows: Sequence[Mapping[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    counter = Counter(int(row["action"]) for row in rows)
    return [{"action": action, "count": int(count)} for action, count in counter.most_common(limit)]


def _train_row_count(dataset: ReplayTrajectoryDataset, episodes: Sequence[int]) -> int:
    if not episodes:
        return 0
    return int(np.count_nonzero(dataset.policy_train_mask[:, list(episodes)]))


def _state_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    row_index = int(step_index) * int(dataset.episode_count) + int(episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    return _hash_arrays(
        np.asarray(dataset.obs[step_index, episode_index]),
        np.asarray(
            [dataset.actor[step_index, episode_index], dataset.to_play_seat[step_index, episode_index]],
            dtype=np.int64,
        ),
        np.asarray(dataset.legal_ids[start:stop], dtype=np.uint32),
    )


def _history_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    stop = int(step_index) + 1
    return _hash_arrays(
        np.asarray(dataset.obs[:stop, episode_index]),
        np.asarray(dataset.actor[:stop, episode_index], dtype=np.int64),
        np.asarray(dataset.to_play_seat[:stop, episode_index], dtype=np.int64),
        np.asarray(dataset.reset_before_step[:stop, episode_index], dtype=np.bool_),
    )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_optional_action_catalog(spec_bundle_json: Path | None) -> ActionCatalog | None:
    if spec_bundle_json is None:
        return None
    spec_bundle = load_json_object(Path(spec_bundle_json), label="spec bundle")
    return ActionCatalog.from_spec_bundle(spec_bundle)


def _decoded_action(action_id: int, catalog: ActionCatalog | None) -> dict[str, Any]:
    if catalog is None:
        return {"action_id": int(action_id), "family": None}
    return _jsonable(asdict(catalog.decode(int(action_id))))


def _action_family(action_id: int, catalog: ActionCatalog | None) -> str | None:
    if catalog is None:
        return None
    return catalog.decode(int(action_id)).family


def _nonempty_episode_metadata_count(bundles: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for bundle in bundles if str(bundle.get(key) or "").strip())


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_optional_int(bundle: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _optional_int(bundle.get(key))
        if value is not None:
            return value
    return None


def _jsonable(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "PairedOutcomePreferenceDecisionConfig",
    "build_paired_outcome_preference_decision_report",
    "write_paired_outcome_preference_decision_report",
]
