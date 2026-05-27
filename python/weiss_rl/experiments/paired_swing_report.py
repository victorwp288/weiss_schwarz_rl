from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS

OPPONENT_TAGS = ("fixed", "learned", "champion", "recent", "hard_negative", "other")
GROUP_NAMES = ("all", *OPPONENT_TAGS)


@dataclass(frozen=True, slots=True)
class PairedSwingReportConfig:
    compare_jsons: tuple[Path, ...]
    opponent_pool_jsonls: tuple[Path, ...] = ()
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_opponents: tuple[str, ...] = ()
    max_examples_per_bucket: int = 24
    notes: str = ""


def build_paired_swing_report(config: PairedSwingReportConfig) -> dict[str, Any]:
    fixed = set(config.fixed_opponents)
    learned = set(config.learned_opponents)
    pool_tags = _load_pool_tags(config.opponent_pool_jsonls)
    compare_reports = [_read_json_object(path) for path in config.compare_jsons]

    comparisons = [
        _summarize_comparison(
            payload=payload,
            compare_json=path,
            fixed_opponents=fixed,
            learned_opponents=learned,
            pool_tags=pool_tags,
            max_examples_per_bucket=max(0, int(config.max_examples_per_bucket)),
        )
        for path, payload in zip(config.compare_jsons, compare_reports, strict=True)
    ]
    aggregate = _aggregate_comparisons(comparisons)
    return {
        "kind": "paired_swing_report_v1",
        "compare_jsons": [path.as_posix() for path in config.compare_jsons],
        "opponent_pool_jsonls": [path.as_posix() for path in config.opponent_pool_jsonls],
        "fixed_opponents": list(config.fixed_opponents),
        "learned_opponents": list(config.learned_opponents),
        "pool_tags": {opponent: sorted(tags) for opponent, tags in sorted(pool_tags.items())},
        "notes": str(config.notes),
        "aggregate": aggregate,
        "comparisons": comparisons,
    }


def write_paired_swing_report(report: Mapping[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_comparison(
    *,
    payload: Mapping[str, Any],
    compare_json: Path,
    fixed_opponents: set[str],
    learned_opponents: set[str],
    pool_tags: Mapping[str, set[str]],
    max_examples_per_bucket: int,
) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"paired compare JSON is missing rows: {compare_json}")

    group_summaries = _empty_group_summaries()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    opponent_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        opponent = str(row.get("opponent_policy_id", ""))
        if not opponent:
            continue
        tags = _opponent_tags(
            opponent,
            fixed_opponents=fixed_opponents,
            learned_opponents=learned_opponents,
            pool_tags=pool_tags,
        )
        compact_row = _compact_row(row, tags=tags)
        opponent_rows.append(compact_row)
        _add_row_to_groups(group_summaries, compact_row)
        for example in row.get("examples", []):
            if not isinstance(example, Mapping):
                continue
            repair_target = _repair_target(
                example,
                opponent=opponent,
                tags=tags,
                compare_json=compare_json,
                baseline_label=_label(payload.get("baseline"), default="baseline"),
                candidate_label=_label(payload.get("candidate"), default="candidate"),
            )
            if repair_target is None:
                continue
            bucket_key = _bucket_key(repair_target)
            if len(buckets[bucket_key]) < max_examples_per_bucket:
                buckets[bucket_key].append(repair_target)

    return {
        "compare_json": compare_json.as_posix(),
        "baseline": payload.get("baseline", {}),
        "candidate": payload.get("candidate", {}),
        "groups": _finalize_groups(group_summaries),
        "opponent_rows": sorted(opponent_rows, key=_row_sort_key),
        "repair_targets": dict(sorted(buckets.items())),
        "replay_seed_plan": _replay_seed_plan(buckets),
    }


def _aggregate_comparisons(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = _empty_group_summaries()
    for comparison in comparisons:
        comparison_groups = comparison.get("groups")
        if not isinstance(comparison_groups, Mapping):
            continue
        for name in GROUP_NAMES:
            group = comparison_groups.get(name)
            if isinstance(group, Mapping):
                _add_group(groups[name], group, opponent_increment=_int(group.get("opponent_count")))
    return {"groups": _finalize_groups(groups)}


def _compact_row(row: Mapping[str, Any], *, tags: set[str]) -> dict[str, Any]:
    shared_games = _int(row.get("shared_games"))
    baseline_wins = _int(row.get("baseline_wins"))
    candidate_wins = _int(row.get("candidate_wins"))
    return {
        "opponent_policy_id": str(row.get("opponent_policy_id", "")),
        "tags": sorted(tags),
        "shared_games": shared_games,
        "baseline_wins": baseline_wins,
        "candidate_wins": candidate_wins,
        "delta_wins": candidate_wins - baseline_wins,
        "candidate_gain_flips": _int(row.get("baseline_nonwin_candidate_win")),
        "candidate_regression_flips": _int(row.get("baseline_win_candidate_nonwin")),
        "changed_outcome": _int(row.get("changed_outcome")),
        "mean_decision_count_delta": row.get("mean_decision_count_delta"),
        "mean_pass_actions_delta": row.get("mean_pass_actions_delta"),
        "mean_pass_with_nonpass_available_delta": row.get("mean_pass_with_nonpass_available_delta"),
    }


def _repair_target(
    example: Mapping[str, Any],
    *,
    opponent: str,
    tags: set[str],
    compare_json: Path,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any] | None:
    baseline_win = _is_win(example.get("baseline_outcome"))
    candidate_win = _is_win(example.get("candidate_outcome"))
    if baseline_win and not candidate_win:
        direction = "regression"
    elif not baseline_win and candidate_win:
        direction = "gain"
    else:
        direction = "changed_nonbinary"

    return {
        "direction": direction,
        "opponent_policy_id": opponent,
        "tags": sorted(tags),
        "pair_index": _int(example.get("pair_index")),
        "swap_index": _int(example.get("swap_index")),
        "episode_seed": _int(example.get("episode_seed")),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_outcome": example.get("baseline_outcome"),
        "candidate_outcome": example.get("candidate_outcome"),
        "decision_count_delta": _int(example.get("candidate_decision_count"))
        - _int(example.get("baseline_decision_count")),
        "pass_actions_delta": _int(example.get("candidate_pass_actions")) - _int(example.get("baseline_pass_actions")),
        "pass_with_nonpass_available_delta": _int(example.get("candidate_pass_with_nonpass_available"))
        - _int(example.get("baseline_pass_with_nonpass_available")),
        "source_compare_json": compare_json.as_posix(),
    }


def _bucket_key(target: Mapping[str, Any]) -> str:
    direction = str(target.get("direction", "unknown"))
    tags = set(str(tag) for tag in target.get("tags", []))
    if "fixed" in tags:
        group = "fixed"
    elif "hard_negative" in tags:
        group = "hard_negative"
    elif "champion" in tags:
        group = "champion"
    elif "learned" in tags:
        group = "learned"
    else:
        group = "other"
    return f"{group}_{direction}"


def _replay_seed_plan(buckets: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, list[int]]]:
    plan: dict[str, dict[str, list[int]]] = {}
    for bucket, targets in buckets.items():
        by_opponent: dict[str, set[int]] = defaultdict(set)
        for target in targets:
            opponent = str(target.get("opponent_policy_id", ""))
            seed = _int(target.get("episode_seed"))
            if opponent and seed:
                by_opponent[opponent].add(seed)
        plan[str(bucket)] = {opponent: sorted(seeds) for opponent, seeds in sorted(by_opponent.items())}
    return dict(sorted(plan.items()))


def _opponent_tags(
    opponent: str,
    *,
    fixed_opponents: set[str],
    learned_opponents: set[str],
    pool_tags: Mapping[str, set[str]],
) -> set[str]:
    tags = _matching_pool_tags(opponent, pool_tags)
    if opponent in fixed_opponents:
        tags.add("fixed")
    if _matches_any_seed_wrapped_suffix(opponent, learned_opponents) or opponent.startswith("seed_"):
        tags.add("learned")
    if not tags:
        tags.add("other")
    return tags


def _matching_pool_tags(opponent: str, pool_tags: Mapping[str, set[str]]) -> set[str]:
    tags: set[str] = set(pool_tags.get(opponent, set()))
    for pool_opponent, pool_opponent_tags in pool_tags.items():
        if _is_seed_wrapped_suffix_match(opponent, pool_opponent):
            tags.update(pool_opponent_tags)
    return tags


def _matches_any_seed_wrapped_suffix(opponent: str, candidates: set[str]) -> bool:
    if opponent in candidates:
        return True
    return any(_is_seed_wrapped_suffix_match(opponent, candidate) for candidate in candidates)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _load_pool_tags(paths: Iterable[Path]) -> dict[str, set[str]]:
    tags: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"opponent pool log not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
                if not isinstance(record, Mapping):
                    continue
                _add_tagged_ids(tags, record.get("champion_ids"), "champion")
                _add_tagged_ids(tags, record.get("recent_ids"), "recent")
                _add_tagged_ids(tags, record.get("hard_negative_ids"), "hard_negative")
    return dict(tags)


def _add_tagged_ids(tags: dict[str, set[str]], values: Any, tag: str) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        if isinstance(value, str) and value:
            tags[value].add(tag)


def _empty_group_summaries() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "opponent_count": 0,
            "shared_games": 0,
            "baseline_wins": 0,
            "candidate_wins": 0,
            "delta_wins": 0,
            "candidate_gain_flips": 0,
            "candidate_regression_flips": 0,
            "changed_outcome": 0,
        }
        for name in GROUP_NAMES
    }


def _add_row_to_groups(groups: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    _add_group(groups["all"], row, opponent_increment=1)
    tags = set(str(tag) for tag in row.get("tags", []))
    for tag in OPPONENT_TAGS:
        if tag in tags:
            _add_group(groups[tag], row, opponent_increment=1)


def _add_group(target: dict[str, Any], source: Mapping[str, Any], opponent_increment: int = 0) -> None:
    target["opponent_count"] = _int(target.get("opponent_count")) + opponent_increment
    for key in (
        "shared_games",
        "baseline_wins",
        "candidate_wins",
        "delta_wins",
        "candidate_gain_flips",
        "candidate_regression_flips",
        "changed_outcome",
    ):
        target[key] = _int(target.get(key)) + _int(source.get(key))


def _finalize_groups(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for name, group in groups.items():
        shared_games = _int(group.get("shared_games"))
        delta_wins = _int(group.get("delta_wins"))
        finalized[name] = {
            **dict(group),
            "baseline_mean": None if shared_games <= 0 else _int(group.get("baseline_wins")) / shared_games,
            "candidate_mean": None if shared_games <= 0 else _int(group.get("candidate_wins")) / shared_games,
            "delta_mean": None if shared_games <= 0 else delta_wins / shared_games,
        }
    return finalized


def _label(value: Any, *, default: str) -> str:
    if isinstance(value, Mapping):
        label = value.get("label")
        if isinstance(label, str) and label:
            return label
    return default


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (-abs(_int(row.get("delta_wins"))), str(row.get("opponent_policy_id", "")))


def _is_win(value: Any) -> bool:
    return str(value) == "W"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
