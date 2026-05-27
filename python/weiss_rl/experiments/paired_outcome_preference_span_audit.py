"""Trajectory-span diagnostics for paired-outcome preference datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.experiments.paired_outcome_preference_decisions import (
    _aligned_edges_for_pair,
    _load_optional_action_catalog,
    _selected_bundles,
)
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSpanAuditConfig:
    dataset_path: Path
    spec_bundle_json: Path | None = None
    max_gap: int = 1
    max_compact_span_width: int = 8
    min_repeated_pair_count: int = 2
    max_examples: int = 20


def build_paired_outcome_preference_span_audit(
    config: PairedOutcomePreferenceSpanAuditConfig,
) -> dict[str, Any]:
    """Find compact repeated action/family spans in preferred-vs-rejected trajectories."""

    dataset = load_replay_trajectory_bc_dataset(config.dataset_path)
    bundles = _selected_bundles(dataset)
    catalog = _load_optional_action_catalog(config.spec_bundle_json)
    pair_roles = _episodes_by_pair_role(bundles)
    pair_summaries: list[dict[str, Any]] = []
    all_diff_edges: list[dict[str, Any]] = []
    all_spans: list[dict[str, Any]] = []

    for pair_id in sorted(pair_roles):
        by_role = pair_roles[pair_id]
        preferred = list(by_role.get(1, []))
        rejected = list(by_role.get(0, []))
        complete_pair = bool(preferred and rejected)
        pair_edges: list[dict[str, Any]] = []
        diff_edges: list[dict[str, Any]] = []
        spans: list[dict[str, Any]] = []
        if complete_pair:
            pair_edges = _aligned_edges_for_pair(
                dataset,
                bundles=bundles,
                catalog=catalog,
                preference_pair_id=pair_id,
                preferred_episode=int(preferred[0]),
                rejected_episode=int(rejected[0]),
            )
            diff_edges = [_annotate_edge(edge) for edge in pair_edges if not bool(edge.get("same_action"))]
            spans = _spans_for_diff_edges(
                diff_edges,
                max_gap=int(config.max_gap),
                max_compact_span_width=int(config.max_compact_span_width),
            )
            all_diff_edges.extend(diff_edges)
            all_spans.extend(spans)

        bundle = bundles[preferred[0]] if preferred else (bundles[rejected[0]] if rejected else {})
        pair_summary = {
            "preference_pair_id": int(pair_id),
            "complete_pair": complete_pair,
            "preferred_episode_indices": preferred,
            "rejected_episode_indices": rejected,
            "source_opponent_policy_id": str(bundle.get("source_opponent_policy_id") or ""),
            "source_pair_index": _optional_int(bundle.get("source_pair_index", bundle.get("pair_index"))),
            "episode_seed": _optional_int(bundle.get("episode_seed")),
            "swap_index": _optional_int(bundle.get("swap_index")),
            "aligned_step_count": len(pair_edges),
            "different_action_count": len(diff_edges),
            "same_family_different_action_count": sum(1 for edge in diff_edges if bool(edge["same_family"])),
            "same_current_state_different_action_count": sum(
                1 for edge in diff_edges if bool(edge.get("same_current_state"))
            ),
            "same_history_different_action_count": sum(1 for edge in diff_edges if bool(edge.get("same_history"))),
            "span_count": len(spans),
            "compact_span_count": sum(1 for span in spans if bool(span["compact"])),
            "earliest_different_edge": _compact_edge(diff_edges[0]) if diff_edges else None,
            "earliest_span": _compact_span(spans[0], max_edges=int(config.max_examples)) if spans else None,
            "densest_span": _compact_span(_densest_span(spans), max_edges=int(config.max_examples)) if spans else None,
            "family_edge_counts": _edge_count_rows(diff_edges, key_name="family_edge", limit=int(config.max_examples)),
            "action_label_edge_counts": _edge_count_rows(
                diff_edges, key_name="action_label_edge", limit=int(config.max_examples)
            ),
            "raw_action_edge_counts": _edge_count_rows(
                diff_edges, key_name="raw_action_edge", limit=int(config.max_examples)
            ),
        }
        pair_summaries.append(pair_summary)

    repeated_family_edges = _repeated_edge_rows(
        all_diff_edges,
        all_spans,
        key_name="family_edge",
        min_repeated_pair_count=int(config.min_repeated_pair_count),
        limit=int(config.max_examples),
    )
    repeated_action_label_edges = _repeated_edge_rows(
        all_diff_edges,
        all_spans,
        key_name="action_label_edge",
        min_repeated_pair_count=int(config.min_repeated_pair_count),
        limit=int(config.max_examples),
    )
    repeated_raw_action_edges = _repeated_edge_rows(
        all_diff_edges,
        all_spans,
        key_name="raw_action_edge",
        min_repeated_pair_count=int(config.min_repeated_pair_count),
        limit=int(config.max_examples),
    )
    opponent_summaries = _opponent_summaries(
        pair_summaries=pair_summaries,
        diff_edges=all_diff_edges,
        spans=all_spans,
        min_repeated_pair_count=int(config.min_repeated_pair_count),
        max_examples=int(config.max_examples),
    )
    gate = _span_gate(opponent_summaries=opponent_summaries)

    return {
        "kind": "paired_outcome_preference_span_audit_v1",
        "passed": bool(gate["passed"]),
        "failures": list(gate["failures"]),
        "dataset_path": config.dataset_path.as_posix(),
        "spec_bundle_json": None if config.spec_bundle_json is None else config.spec_bundle_json.as_posix(),
        "episode_count": int(dataset.episode_count),
        "train_rows": int(np.count_nonzero(dataset.policy_train_mask)),
        "preference_pair_count": len(pair_summaries),
        "complete_pair_count": sum(1 for row in pair_summaries if bool(row["complete_pair"])),
        "aligned_step_count": sum(int(row["aligned_step_count"]) for row in pair_summaries),
        "different_action_count": len(all_diff_edges),
        "span_count": len(all_spans),
        "compact_span_count": sum(1 for span in all_spans if bool(span["compact"])),
        "summary": {
            "complete_pair_count": sum(1 for row in pair_summaries if bool(row["complete_pair"])),
            "different_action_count": len(all_diff_edges),
            "span_count": len(all_spans),
            "compact_span_count": sum(1 for span in all_spans if bool(span["compact"])),
            "passing_opponents": list(gate["passing_opponents"]),
        },
        "thresholds": {
            "max_gap": int(config.max_gap),
            "max_compact_span_width": int(config.max_compact_span_width),
            "min_repeated_pair_count": int(config.min_repeated_pair_count),
        },
        "span_gate": gate,
        "repeated_family_edges": repeated_family_edges,
        "repeated_action_label_edges": repeated_action_label_edges,
        "repeated_raw_action_edges": repeated_raw_action_edges,
        "candidate_spans": [
            _compact_span(span, max_edges=int(config.max_examples)) for span in all_spans if bool(span["compact"])
        ],
        "opponent_summaries": opponent_summaries,
        "pair_summaries": pair_summaries,
    }


def write_paired_outcome_preference_span_audit(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episodes_by_pair_role(bundles: Sequence[Mapping[str, Any]]) -> dict[int, dict[int, list[int]]]:
    result: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for episode_index, bundle in enumerate(bundles):
        pair_id = _optional_int(bundle.get("preference_pair_id"))
        role = _optional_int(bundle.get("preference_role"))
        if pair_id is None or role not in {0, 1}:
            continue
        result[int(pair_id)][int(role)].append(int(episode_index))
    return result


def _annotate_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    preferred = edge.get("preferred_decoded_action")
    rejected = edge.get("rejected_decoded_action")
    preferred_decoded = preferred if isinstance(preferred, Mapping) else {}
    rejected_decoded = rejected if isinstance(rejected, Mapping) else {}
    preferred_action = int(edge["preferred_action"])
    rejected_action = int(edge["rejected_action"])
    preferred_family = _family(preferred_decoded, preferred_action)
    rejected_family = _family(rejected_decoded, rejected_action)
    annotated = dict(edge)
    annotated.update(
        {
            "preferred_family": preferred_family,
            "rejected_family": rejected_family,
            "family_edge": f"{preferred_family}->{rejected_family}",
            "preferred_action_label": _action_label(preferred_decoded, preferred_action),
            "rejected_action_label": _action_label(rejected_decoded, rejected_action),
            "action_label_edge": (
                f"{_action_label(preferred_decoded, preferred_action)}"
                f"->{_action_label(rejected_decoded, rejected_action)}"
            ),
            "raw_action_edge": f"{preferred_action}->{rejected_action}",
        }
    )
    return annotated


def _spans_for_diff_edges(
    diff_edges: Sequence[Mapping[str, Any]],
    *,
    max_gap: int,
    max_compact_span_width: int,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    for edge in sorted(diff_edges, key=lambda item: int(item["step_index"])):
        if current and int(edge["step_index"]) > int(current[-1]["step_index"]) + max(1, max_gap):
            spans.append(_summarize_span(current, max_compact_span_width=max_compact_span_width))
            current = []
        current.append(edge)
    if current:
        spans.append(_summarize_span(current, max_compact_span_width=max_compact_span_width))
    return spans


def _summarize_span(edges: Sequence[Mapping[str, Any]], *, max_compact_span_width: int) -> dict[str, Any]:
    start_step = int(edges[0]["step_index"])
    end_step = int(edges[-1]["step_index"])
    span_width = end_step - start_step + 1
    family_counts = Counter(str(edge["family_edge"]) for edge in edges)
    action_label_counts = Counter(str(edge["action_label_edge"]) for edge in edges)
    raw_action_counts = Counter(str(edge["raw_action_edge"]) for edge in edges)
    first_edge = edges[0]
    return {
        "preference_pair_id": int(first_edge["preference_pair_id"]),
        "source_opponent_policy_id": str(first_edge.get("source_opponent_policy_id") or ""),
        "source_pair_index": _optional_int(first_edge.get("source_pair_index")),
        "episode_seed": _optional_int(first_edge.get("episode_seed")),
        "swap_index": _optional_int(first_edge.get("swap_index")),
        "start_step": start_step,
        "end_step": end_step,
        "span_width": span_width,
        "different_action_count": len(edges),
        "edge_step_indices": sorted(int(edge["step_index"]) for edge in edges),
        "density": len(edges) / max(span_width, 1),
        "compact": span_width <= int(max_compact_span_width),
        "same_family_count": sum(1 for edge in edges if bool(edge["same_family"])),
        "same_current_state_count": sum(1 for edge in edges if bool(edge.get("same_current_state"))),
        "same_history_count": sum(1 for edge in edges if bool(edge.get("same_history"))),
        "primary_family_edge": family_counts.most_common(1)[0][0],
        "primary_action_label_edge": action_label_counts.most_common(1)[0][0],
        "primary_raw_action_edge": raw_action_counts.most_common(1)[0][0],
        "family_edge_counts": _counter_rows(family_counts),
        "action_label_edge_counts": _counter_rows(action_label_counts),
        "raw_action_edge_counts": _counter_rows(raw_action_counts),
        "edges": [_compact_edge(edge) for edge in edges],
    }


def _densest_span(spans: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not spans:
        raise ValueError("spans must not be empty")
    return max(
        spans,
        key=lambda span: (
            int(span.get("different_action_count") or 0),
            float(span.get("density") or 0.0),
            -int(span.get("start_step") or 0),
        ),
    )


def _edge_count_rows(edges: Sequence[Mapping[str, Any]], *, key_name: str, limit: int) -> list[dict[str, Any]]:
    counter = Counter(str(edge.get(key_name) or "") for edge in edges)
    return _counter_rows(counter, limit=limit)


def _repeated_edge_rows(
    edges: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
    min_repeated_pair_count: int,
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    compact_pairs: dict[str, set[int]] = defaultdict(set)
    for edge in edges:
        grouped[str(edge.get(key_name) or "")].append(edge)
    for span in spans:
        if not bool(span.get("compact")):
            continue
        pair_id = _optional_int(span.get("preference_pair_id"))
        if pair_id is None:
            continue
        count_key = f"{key_name}_counts"
        for row in span.get(count_key, []):
            if isinstance(row, Mapping):
                compact_pairs[str(row.get("key") or "")].add(int(pair_id))
    rows: list[dict[str, Any]] = []
    for key, group_edges in grouped.items():
        if not key:
            continue
        pair_ids = sorted({_required_int(edge.get("preference_pair_id")) for edge in group_edges})
        compact_pair_ids = sorted(compact_pairs.get(key, set()))
        if len(compact_pair_ids) < int(min_repeated_pair_count):
            continue
        rows.append(
            {
                "key": key,
                "count": len(group_edges),
                "pair_count": len(pair_ids),
                "compact_pair_count": len(compact_pair_ids),
                "preference_pair_ids": pair_ids,
                "compact_preference_pair_ids": compact_pair_ids,
                "source_pair_indices": sorted(
                    {
                        _required_int(edge.get("source_pair_index"))
                        for edge in group_edges
                        if _optional_int(edge.get("source_pair_index")) is not None
                    }
                ),
                "opponents": sorted({str(edge.get("source_opponent_policy_id") or "") for edge in group_edges}),
                "first_examples": [_compact_edge(edge) for edge in group_edges[: max(0, min(limit, 5))]],
            }
        )
    rows.sort(key=lambda row: (-int(row["compact_pair_count"]), -int(row["count"]), str(row["key"])))
    return rows[: max(0, limit)]


def _opponent_summaries(
    *,
    pair_summaries: Sequence[Mapping[str, Any]],
    diff_edges: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    min_repeated_pair_count: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    pair_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    edge_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    span_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_summaries:
        pair_groups[str(row.get("source_opponent_policy_id") or "")].append(row)
    for edge in diff_edges:
        edge_groups[str(edge.get("source_opponent_policy_id") or "")].append(edge)
    for span in spans:
        span_groups[str(span.get("source_opponent_policy_id") or "")].append(span)

    rows: list[dict[str, Any]] = []
    for opponent in sorted(pair_groups):
        opponent_pairs = pair_groups[opponent]
        opponent_edges = edge_groups.get(opponent, [])
        opponent_spans = span_groups.get(opponent, [])
        required = min(int(min_repeated_pair_count), len(opponent_pairs))
        repeated_action = _repeated_edge_rows(
            opponent_edges,
            opponent_spans,
            key_name="action_label_edge",
            min_repeated_pair_count=required,
            limit=max_examples,
        )
        repeated_family = _repeated_edge_rows(
            opponent_edges,
            opponent_spans,
            key_name="family_edge",
            min_repeated_pair_count=required,
            limit=max_examples,
        )
        rows.append(
            {
                "source_opponent_policy_id": opponent,
                "pair_count": len(opponent_pairs),
                "required_repeated_pair_count": required,
                "different_action_count": len(opponent_edges),
                "span_count": len(opponent_spans),
                "compact_span_count": sum(1 for span in opponent_spans if bool(span.get("compact"))),
                "has_repeated_action_label_edge": bool(repeated_action),
                "has_repeated_family_edge": bool(repeated_family),
                "strongest_repeated_action_label_edges": repeated_action,
                "strongest_repeated_family_edges": repeated_family,
            }
        )
    return rows


def _span_gate(*, opponent_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    passing_groups = [
        row
        for row in opponent_summaries
        if bool(row.get("has_repeated_action_label_edge")) or bool(row.get("has_repeated_family_edge"))
    ]
    if not passing_groups:
        failures.append({"reason": "no_compact_repeated_span_pattern"})
    return {
        "passed": not failures,
        "failures": failures,
        "passing_opponents": [str(row.get("source_opponent_policy_id") or "") for row in passing_groups],
    }


def _compact_span(span: Mapping[str, Any], *, max_edges: int) -> dict[str, Any]:
    compact = {key: value for key, value in span.items() if key != "edges"}
    compact["edges"] = list(span.get("edges", []))[: max(0, max_edges)]
    return compact


def _compact_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": _required_int(edge.get("step_index")),
        "preferred_action": _required_int(edge.get("preferred_action")),
        "rejected_action": _required_int(edge.get("rejected_action")),
        "preferred_action_label": str(edge.get("preferred_action_label") or ""),
        "rejected_action_label": str(edge.get("rejected_action_label") or ""),
        "family_edge": str(edge.get("family_edge") or ""),
        "action_label_edge": str(edge.get("action_label_edge") or ""),
        "raw_action_edge": str(edge.get("raw_action_edge") or ""),
        "same_family": bool(edge.get("same_family")),
        "same_current_state": bool(edge.get("same_current_state")),
        "same_history": bool(edge.get("same_history")),
    }


def _counter_rows(counter: Counter[str], *, limit: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(None if limit is None else max(0, limit))
    return [{"key": str(key), "count": int(count)} for key, count in items if key]


def _family(decoded: Mapping[str, Any], action_id: int) -> str:
    family = decoded.get("family")
    return str(family) if isinstance(family, str) and family else f"action:{int(action_id)}"


def _action_label(decoded: Mapping[str, Any], action_id: int) -> str:
    family = _family(decoded, action_id)
    parts: list[str] = []
    for key in ("hand_index", "stage_slot", "from_slot", "to_slot", "slot", "attack_type", "index"):
        value = decoded.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return family if not parts else f"{family}({', '.join(parts)})"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_int(value: object) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


__all__ = [
    "PairedOutcomePreferenceSpanAuditConfig",
    "build_paired_outcome_preference_span_audit",
    "write_paired_outcome_preference_span_audit",
]
