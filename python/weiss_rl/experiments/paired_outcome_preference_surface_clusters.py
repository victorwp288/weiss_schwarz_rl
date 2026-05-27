"""Public-surface separability diagnostics for paired outcome preference replay."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config import load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.experiments.paired_outcome_preference_decisions import (
    _aligned_edges_for_pair,
    _load_optional_action_catalog,
    _selected_bundles,
)
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSurfaceClusterConfig:
    dataset_path: Path
    spec_bundle_json: Path | None = None
    stack_config_path: Path | None = None
    opponent_context_policy_ids: tuple[str, ...] = ()
    max_examples: int = 25


def build_paired_outcome_preference_surface_cluster_report(
    config: PairedOutcomePreferenceSurfaceClusterConfig,
) -> dict[str, Any]:
    """Classify exact public-state/history preference conflicts by model-input eligibility."""

    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    bundles = _selected_bundles(dataset)
    catalog = _load_optional_action_catalog(config.spec_bundle_json)
    context_policy_ids = _opponent_context_policy_ids(config)
    context_index_by_policy_id = {
        policy_id: index for index, policy_id in enumerate(context_policy_ids, start=1) if policy_id
    }
    edges = _aligned_different_action_edges(dataset, bundles=bundles, catalog=catalog)
    exact_surface_edges = [edge for edge in edges if bool(edge["same_current_state"]) and bool(edge["same_history"])]
    current_only_edges = [
        edge for edge in edges if bool(edge["same_current_state"]) and not bool(edge["same_history"])
    ]
    surface_clusters = _surface_clusters(
        exact_surface_edges,
        context_index_by_policy_id=context_index_by_policy_id,
        max_examples=int(config.max_examples),
    )
    conflict_clusters = [cluster for cluster in surface_clusters if bool(cluster["has_action_conflict"])]
    classification_counts = Counter(str(cluster["classification"]) for cluster in conflict_clusters)
    opponent_summary = _opponent_context_summary(
        edges,
        context_index_by_policy_id=context_index_by_policy_id,
    )
    replay_only_count = int(classification_counts["replay_only_required"])
    context_missing_count = int(classification_counts["opponent_context_required_missing_mapping"])
    context_resolvable_count = int(classification_counts["opponent_context_resolvable"])
    report = {
        "kind": "paired_outcome_preference_surface_cluster_report_v1",
        "dataset_path": config.dataset_path.as_posix(),
        "spec_bundle_json": None if config.spec_bundle_json is None else config.spec_bundle_json.as_posix(),
        "stack_config_path": None if config.stack_config_path is None else config.stack_config_path.as_posix(),
        "episode_count": int(dataset.episode_count),
        "train_rows": int(np.count_nonzero(dataset.policy_train_mask)),
        "aligned_different_action_count": len(edges),
        "same_public_surface_different_action_count": len(exact_surface_edges),
        "same_current_state_different_history_count": len(current_only_edges),
        "surface_cluster_count": len(surface_clusters),
        "surface_conflict_count": len(conflict_clusters),
        "opponent_context_resolvable_conflict_count": context_resolvable_count,
        "opponent_context_required_missing_mapping_count": context_missing_count,
        "replay_only_required_conflict_count": replay_only_count,
        "public_surface_separable": replay_only_count == 0 and context_missing_count == 0,
        "unconditioned_replay_safe": len(conflict_clusters) == 0,
        "opponent_context_conditioning_required": bool(context_resolvable_count or context_missing_count),
        "surface_hash_definition": {
            "current_state_hash": ["obs[step, episode]", "actor", "to_play_seat", "legal_ids"],
            "history_hash": ["obs[:step+1, episode]", "actor[:step+1]", "to_play_seat[:step+1]", "reset_before_step"],
            "legal_signature_hash": ["legal_ids"],
        },
        "model_input_fields": [
            "obs/current_state",
            "actor",
            "to_play_seat",
            "legal_ids",
            "history/recurrent_state",
            "source_opponent_policy_id when opponent-context is enabled",
        ],
        "diagnostic_only_fields": [
            "preference_pair_id",
            "source_pair_index",
            "episode_seed",
            "swap_index",
        ],
        "opponent_context_policy_ids": list(context_policy_ids),
        "opponent_context_summary": opponent_summary,
        "classification_counts": dict(sorted(classification_counts.items())),
        "surface_conflicts": conflict_clusters[: max(0, int(config.max_examples))],
        "surface_clusters": surface_clusters[: max(0, int(config.max_examples))],
        "same_current_state_different_history_examples": [
            _jsonable(edge) for edge in current_only_edges[: max(0, int(config.max_examples))]
        ],
    }
    return report


def write_paired_outcome_preference_surface_cluster_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _aligned_different_action_edges(
    dataset: ReplayTrajectoryDataset,
    *,
    bundles: Sequence[Mapping[str, Any]],
    catalog: ActionCatalog | None,
) -> list[dict[str, Any]]:
    pair_roles: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for episode_index, bundle in enumerate(bundles):
        pair_id = _optional_int(bundle.get("preference_pair_id"))
        role = _optional_int(bundle.get("preference_role"))
        if pair_id is None or role not in {0, 1}:
            continue
        pair_roles[int(pair_id)][int(role)].append(int(episode_index))

    edges: list[dict[str, Any]] = []
    for pair_id in sorted(pair_roles):
        by_role = pair_roles[pair_id]
        preferred = list(by_role.get(1, []))
        rejected = list(by_role.get(0, []))
        if not preferred or not rejected:
            continue
        for edge in _aligned_edges_for_pair(
            dataset,
            bundles=bundles,
            catalog=catalog,
            preference_pair_id=pair_id,
            preferred_episode=int(preferred[0]),
            rejected_episode=int(rejected[0]),
        ):
            if int(edge["preferred_action"]) == int(edge["rejected_action"]):
                continue
            enriched = dict(edge)
            enriched["legal_signature_hash"] = _legal_signature_hash(
                dataset,
                step_index=int(edge["step_index"]),
                episode_index=int(edge["preferred_episode_index"]),
            )
            edges.append(enriched)
    return edges


def _surface_clusters(
    edges: Sequence[Mapping[str, Any]],
    *,
    context_index_by_policy_id: Mapping[str, int],
    max_examples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        current_state_hash = str(edge.get("current_state_hash") or "")
        history_hash = str(edge.get("history_hash") or "")
        if current_state_hash and history_hash:
            grouped[(current_state_hash, history_hash)].append(edge)

    clusters: list[dict[str, Any]] = []
    for (current_state_hash, history_hash), group in sorted(grouped.items()):
        directions = Counter((int(edge["preferred_action"]), int(edge["rejected_action"])) for edge in group)
        exact_reverse_count = _exact_reverse_count(group)
        preferred_actions = sorted({int(edge["preferred_action"]) for edge in group})
        has_action_conflict = bool(exact_reverse_count > 0 or len(preferred_actions) > 1)
        opponent_rows = _surface_opponent_rows(group, context_index_by_policy_id=context_index_by_policy_id)
        same_opponent_exact_reverse_count = sum(int(row["exact_reverse_pair_count"]) for row in opponent_rows)
        missing_context_ids = sorted(
            {
                str(edge.get("source_opponent_policy_id") or "")
                for edge in group
                if _context_index_for_policy_id(
                    str(edge.get("source_opponent_policy_id") or ""),
                    context_index_by_policy_id,
                )
                <= 0
            }
        )
        classification = _classify_surface_conflict(
            has_action_conflict=has_action_conflict,
            same_opponent_exact_reverse_count=same_opponent_exact_reverse_count,
            missing_context_ids=missing_context_ids,
        )
        clusters.append(
            {
                "current_state_hash": current_state_hash,
                "history_hash": history_hash,
                "edge_count": len(group),
                "has_action_conflict": has_action_conflict,
                "classification": classification,
                "exact_reverse_pair_count": exact_reverse_count,
                "same_opponent_exact_reverse_pair_count": same_opponent_exact_reverse_count,
                "preferred_actions": preferred_actions,
                "rejected_actions": sorted({int(edge["rejected_action"]) for edge in group}),
                "action_directions": [
                    {
                        "preferred_action": int(preferred_action),
                        "rejected_action": int(rejected_action),
                        "count": int(count),
                    }
                    for (preferred_action, rejected_action), count in sorted(directions.items())
                ],
                "opponent_policy_ids": sorted({str(edge.get("source_opponent_policy_id") or "") for edge in group}),
                "missing_context_opponent_policy_ids": missing_context_ids,
                "opponent_rows": opponent_rows,
                "source_pair_indices": sorted(
                    {
                        int(edge["source_pair_index"])
                        for edge in group
                        if edge.get("source_pair_index") is not None
                    }
                ),
                "examples": [_jsonable(edge) for edge in group[: max(0, max_examples)]],
            }
        )
    return clusters


def _surface_opponent_rows(
    group: Sequence[Mapping[str, Any]],
    *,
    context_index_by_policy_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in group:
        grouped[str(edge.get("source_opponent_policy_id") or "")].append(edge)
    rows: list[dict[str, Any]] = []
    for opponent_id, opponent_edges in sorted(grouped.items()):
        directions = Counter((int(edge["preferred_action"]), int(edge["rejected_action"])) for edge in opponent_edges)
        rows.append(
            {
                "opponent_policy_id": opponent_id,
                "opponent_context_index": _context_index_for_policy_id(opponent_id, context_index_by_policy_id),
                "edge_count": len(opponent_edges),
                "preferred_actions": sorted({int(edge["preferred_action"]) for edge in opponent_edges}),
                "rejected_actions": sorted({int(edge["rejected_action"]) for edge in opponent_edges}),
                "exact_reverse_pair_count": _exact_reverse_count(opponent_edges),
                "action_directions": [
                    {
                        "preferred_action": int(preferred_action),
                        "rejected_action": int(rejected_action),
                        "count": int(count),
                    }
                    for (preferred_action, rejected_action), count in sorted(directions.items())
                ],
            }
        )
    return rows


def _classify_surface_conflict(
    *,
    has_action_conflict: bool,
    same_opponent_exact_reverse_count: int,
    missing_context_ids: Sequence[str],
) -> str:
    if not has_action_conflict:
        return "unconflicted"
    if same_opponent_exact_reverse_count > 0:
        return "replay_only_required"
    if missing_context_ids:
        return "opponent_context_required_missing_mapping"
    return "opponent_context_resolvable"


def _exact_reverse_count(edges: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for left_index, left in enumerate(edges):
        for right in edges[left_index + 1 :]:
            if int(left["preferred_action"]) == int(right["rejected_action"]) and int(
                left["rejected_action"]
            ) == int(right["preferred_action"]):
                count += 1
    return count


def _opponent_context_summary(
    edges: Sequence[Mapping[str, Any]],
    *,
    context_index_by_policy_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: Counter[str] = Counter(str(edge.get("source_opponent_policy_id") or "") for edge in edges)
    return [
        {
            "opponent_policy_id": opponent_id,
            "edge_count": int(edge_count),
            "opponent_context_index": _context_index_for_policy_id(opponent_id, context_index_by_policy_id),
        }
        for opponent_id, edge_count in sorted(grouped.items())
    ]


def _opponent_context_policy_ids(config: PairedOutcomePreferenceSurfaceClusterConfig) -> tuple[str, ...]:
    policy_ids = [str(policy_id).strip() for policy_id in config.opponent_context_policy_ids if str(policy_id).strip()]
    if config.stack_config_path is not None:
        stack = load_stack_config(config.stack_config_path)
        policy_ids.extend(str(policy_id).strip() for policy_id in stack.config.model.opponent_context_policy_ids)
    result: list[str] = []
    seen: set[str] = set()
    for policy_id in policy_ids:
        if policy_id and policy_id not in seen:
            seen.add(policy_id)
            result.append(policy_id)
    return tuple(result)


def _context_index_for_policy_id(policy_id: str, context_index_by_policy_id: Mapping[str, int]) -> int:
    policy_text = str(policy_id).strip()
    if not policy_text:
        return 0
    exact = context_index_by_policy_id.get(policy_text)
    if exact is not None:
        return int(exact)
    for configured_policy_id, configured_index in context_index_by_policy_id.items():
        if policy_text.endswith(f"_{configured_policy_id}"):
            return int(configured_index)
    return 0


def _legal_signature_hash(dataset: ReplayTrajectoryDataset, *, step_index: int, episode_index: int) -> str:
    row_index = int(step_index) * int(dataset.episode_count) + int(episode_index)
    start = int(dataset.legal_offsets[row_index])
    stop = int(dataset.legal_offsets[row_index + 1])
    digest = hashlib.sha256()
    legal_ids = np.ascontiguousarray(dataset.legal_ids[start:stop], dtype=np.uint32)
    digest.update(str(legal_ids.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(int(item) for item in legal_ids.shape)).encode("ascii"))
    digest.update(legal_ids.tobytes())
    return digest.hexdigest()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
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
    "PairedOutcomePreferenceSurfaceClusterConfig",
    "build_paired_outcome_preference_surface_cluster_report",
    "write_paired_outcome_preference_surface_cluster_report",
]
