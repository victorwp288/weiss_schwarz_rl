from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.eval.export import load_eval_game_records
from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


@dataclass(frozen=True, slots=True)
class PairedOutcomeCompareConfig:
    baseline_summary_json: Path
    candidate_summary_json: Path
    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_opponents: tuple[str, ...] = ()
    max_examples: int = 50
    pair_index_split: int | None = None


def compare_paired_targeted_outcomes(config: PairedOutcomeCompareConfig) -> dict[str, Any]:
    baseline_rows = _load_summary_rows(config.baseline_summary_json)
    candidate_rows = _load_summary_rows(config.candidate_summary_json)
    learned_opponents = config.learned_opponents or _infer_learned_opponents(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        fixed_opponents=config.fixed_opponents,
    )
    opponent_ids = tuple(dict.fromkeys([*config.fixed_opponents, *learned_opponents]))
    compared_opponents = opponent_ids or tuple(sorted(set(baseline_rows) & set(candidate_rows)))
    rows = []
    for opponent in compared_opponents:
        baseline_match = _resolve_opponent_row(baseline_rows, opponent)
        candidate_match = _resolve_opponent_row(candidate_rows, opponent)
        rows.append(
            _compare_opponent(
                opponent=opponent,
                baseline_summary_json=config.baseline_summary_json,
                candidate_summary_json=config.candidate_summary_json,
                baseline_row=baseline_match.row,
                candidate_row=candidate_match.row,
                baseline_opponent_policy_id=baseline_match.opponent_policy_id,
                candidate_opponent_policy_id=candidate_match.opponent_policy_id,
                baseline_ambiguous_opponent_policy_ids=baseline_match.ambiguous_opponent_policy_ids,
                candidate_ambiguous_opponent_policy_ids=candidate_match.ambiguous_opponent_policy_ids,
                max_examples=int(config.max_examples),
                pair_index_split=config.pair_index_split,
            )
        )
    groups = {
        "fixed_baselines": _summarize_group(rows, opponents=config.fixed_opponents),
        "learned_opponents": _summarize_group(rows, opponents=learned_opponents),
        "all_compared": _summarize_group(rows, opponents=compared_opponents),
    }
    groups_by_pair_index_bucket = (
        {
            "fixed_baselines": _summarize_group_pair_buckets(rows, opponents=config.fixed_opponents),
            "learned_opponents": _summarize_group_pair_buckets(rows, opponents=learned_opponents),
            "all_compared": _summarize_group_pair_buckets(rows, opponents=compared_opponents),
        }
        if config.pair_index_split is not None
        else {}
    )
    return {
        "kind": "paired_targeted_outcome_compare_v1",
        "baseline": {
            "label": str(config.baseline_label),
            "summary_json": config.baseline_summary_json.as_posix(),
        },
        "candidate": {
            "label": str(config.candidate_label),
            "summary_json": config.candidate_summary_json.as_posix(),
        },
        "fixed_opponents": list(config.fixed_opponents),
        "learned_opponents": list(learned_opponents),
        "learned_opponents_inferred": not bool(config.learned_opponents),
        "pair_index_split": config.pair_index_split,
        "rows": rows,
        "groups": groups,
        "groups_by_pair_index_bucket": groups_by_pair_index_bucket,
    }


def _infer_learned_opponents(
    *,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    fixed_opponents: Sequence[str],
) -> tuple[str, ...]:
    fixed = tuple(str(opponent) for opponent in fixed_opponents)
    inferred: list[str] = []
    for opponent in candidate_rows:
        if any(_is_seed_wrapped_suffix_match(str(opponent), fixed_opponent) for fixed_opponent in fixed):
            continue
        if _resolve_opponent_row(baseline_rows, str(opponent)).row is None:
            continue
        inferred.append(str(opponent))
    return tuple(dict.fromkeys(inferred))


@dataclass(frozen=True, slots=True)
class _ResolvedOpponentRow:
    row: Mapping[str, Any] | None
    opponent_policy_id: str | None
    ambiguous_opponent_policy_ids: tuple[str, ...] = ()


def _resolve_opponent_row(rows: Mapping[str, Mapping[str, Any]], opponent: str) -> _ResolvedOpponentRow:
    row = rows.get(opponent)
    if row is not None:
        return _ResolvedOpponentRow(row=row, opponent_policy_id=opponent)

    suffix_matches = tuple(
        row_opponent for row_opponent in rows if _is_seed_wrapped_suffix_match(str(row_opponent), str(opponent))
    )
    if len(suffix_matches) == 1:
        resolved = suffix_matches[0]
        return _ResolvedOpponentRow(row=rows[resolved], opponent_policy_id=resolved)
    if len(suffix_matches) > 1:
        return _ResolvedOpponentRow(
            row=None,
            opponent_policy_id=None,
            ambiguous_opponent_policy_ids=tuple(sorted(suffix_matches)),
        )
    return _ResolvedOpponentRow(row=None, opponent_policy_id=None)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _compare_opponent(
    *,
    opponent: str,
    baseline_summary_json: Path,
    candidate_summary_json: Path,
    baseline_row: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any] | None,
    baseline_opponent_policy_id: str | None,
    candidate_opponent_policy_id: str | None,
    baseline_ambiguous_opponent_policy_ids: Sequence[str],
    candidate_ambiguous_opponent_policy_ids: Sequence[str],
    max_examples: int,
    pair_index_split: int | None,
) -> dict[str, Any]:
    if baseline_row is None or candidate_row is None:
        return {
            "opponent_policy_id": opponent,
            "status": "missing",
            "missing_baseline": baseline_row is None,
            "missing_candidate": candidate_row is None,
            "baseline_opponent_policy_id": baseline_opponent_policy_id,
            "candidate_opponent_policy_id": candidate_opponent_policy_id,
            "baseline_ambiguous_opponent_policy_ids": list(baseline_ambiguous_opponent_policy_ids),
            "candidate_ambiguous_opponent_policy_ids": list(candidate_ambiguous_opponent_policy_ids),
        }

    baseline_records = _load_row_records(baseline_row, summary_json=baseline_summary_json)
    candidate_records = _load_row_records(candidate_row, summary_json=candidate_summary_json)
    baseline_by_key = {_record_key(record): record for record in baseline_records}
    candidate_by_key = {_record_key(record): record for record in candidate_records}
    shared_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    missing_baseline = sorted(set(candidate_by_key) - set(baseline_by_key))
    missing_candidate = sorted(set(baseline_by_key) - set(candidate_by_key))

    baseline_wins = 0
    candidate_wins = 0
    baseline_win_candidate_nonwin = 0
    baseline_nonwin_candidate_win = 0
    same_win = 0
    same_nonwin = 0
    changed_outcome = 0
    candidate_decision_delta_total = 0
    candidate_pass_delta_total = 0
    candidate_pass_nonpass_delta_total = 0
    examples: list[dict[str, Any]] = []
    pair_bucket_stats: dict[str, dict[str, int]] = {}

    for key in shared_keys:
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        baseline_score = _win_score(baseline)
        candidate_score = _win_score(candidate)
        baseline_wins += baseline_score
        candidate_wins += candidate_score
        if baseline_score == 1 and candidate_score == 0:
            baseline_win_candidate_nonwin += 1
        elif baseline_score == 0 and candidate_score == 1:
            baseline_nonwin_candidate_win += 1
        elif baseline_score == 1 and candidate_score == 1:
            same_win += 1
        else:
            same_nonwin += 1
        if baseline.outcome != candidate.outcome:
            changed_outcome += 1
            if len(examples) < max(0, max_examples):
                examples.append(_flip_example(baseline, candidate))
        candidate_decision_delta_total += int(candidate.decision_count) - int(baseline.decision_count)
        candidate_pass_delta_total += int(candidate.pass_actions) - int(baseline.pass_actions)
        candidate_pass_nonpass_delta_total += int(candidate.pass_with_nonpass_available) - int(
            baseline.pass_with_nonpass_available
        )
        if pair_index_split is not None:
            bucket_name = _pair_index_bucket_name(int(baseline.pair_index), split=int(pair_index_split))
            bucket = pair_bucket_stats.setdefault(bucket_name, _empty_int_stats())
            bucket["shared_games"] += 1
            bucket["baseline_wins"] += baseline_score
            bucket["candidate_wins"] += candidate_score
            if baseline_score == 1 and candidate_score == 0:
                bucket["baseline_win_candidate_nonwin"] += 1
            elif baseline_score == 0 and candidate_score == 1:
                bucket["baseline_nonwin_candidate_win"] += 1
            if baseline.outcome != candidate.outcome:
                bucket["changed_outcome"] += 1

    shared_games = len(shared_keys)
    return {
        "opponent_policy_id": opponent,
        "baseline_opponent_policy_id": baseline_opponent_policy_id,
        "candidate_opponent_policy_id": candidate_opponent_policy_id,
        "status": "ok",
        "shared_games": shared_games,
        "baseline_games": len(baseline_records),
        "candidate_games": len(candidate_records),
        "missing_baseline_games": len(missing_baseline),
        "missing_candidate_games": len(missing_candidate),
        "baseline_wins": baseline_wins,
        "candidate_wins": candidate_wins,
        "delta_wins": candidate_wins - baseline_wins,
        "baseline_mean": None if shared_games <= 0 else baseline_wins / shared_games,
        "candidate_mean": None if shared_games <= 0 else candidate_wins / shared_games,
        "delta_mean": None if shared_games <= 0 else (candidate_wins - baseline_wins) / shared_games,
        "baseline_win_candidate_nonwin": baseline_win_candidate_nonwin,
        "baseline_nonwin_candidate_win": baseline_nonwin_candidate_win,
        "same_win": same_win,
        "same_nonwin": same_nonwin,
        "changed_outcome": changed_outcome,
        "mean_decision_count_delta": None
        if shared_games <= 0
        else candidate_decision_delta_total / float(shared_games),
        "mean_pass_actions_delta": None if shared_games <= 0 else candidate_pass_delta_total / float(shared_games),
        "mean_pass_with_nonpass_available_delta": None
        if shared_games <= 0
        else candidate_pass_nonpass_delta_total / float(shared_games),
        "pair_index_buckets": _finalize_pair_bucket_stats(pair_bucket_stats) if pair_index_split is not None else {},
        "examples": examples,
    }


def _summarize_group(rows: Sequence[Mapping[str, Any]], *, opponents: Sequence[str]) -> dict[str, Any]:
    opponent_set = set(opponents)
    selected = [row for row in rows if str(row.get("opponent_policy_id")) in opponent_set]
    shared_games = sum(int(row.get("shared_games", 0)) for row in selected if row.get("status") == "ok")
    baseline_wins = sum(int(row.get("baseline_wins", 0)) for row in selected if row.get("status") == "ok")
    candidate_wins = sum(int(row.get("candidate_wins", 0)) for row in selected if row.get("status") == "ok")
    return {
        "opponents": list(opponents),
        "present_count": sum(1 for row in selected if row.get("status") == "ok"),
        "expected_count": len(tuple(opponents)),
        "shared_games": shared_games,
        "baseline_wins": baseline_wins,
        "candidate_wins": candidate_wins,
        "delta_wins": candidate_wins - baseline_wins,
        "baseline_mean": None if shared_games <= 0 else baseline_wins / shared_games,
        "candidate_mean": None if shared_games <= 0 else candidate_wins / shared_games,
        "delta_mean": None if shared_games <= 0 else (candidate_wins - baseline_wins) / shared_games,
        "baseline_win_candidate_nonwin": sum(
            int(row.get("baseline_win_candidate_nonwin", 0)) for row in selected if row.get("status") == "ok"
        ),
        "baseline_nonwin_candidate_win": sum(
            int(row.get("baseline_nonwin_candidate_win", 0)) for row in selected if row.get("status") == "ok"
        ),
        "changed_outcome": sum(int(row.get("changed_outcome", 0)) for row in selected if row.get("status") == "ok"),
    }


def _summarize_group_pair_buckets(
    rows: Sequence[Mapping[str, Any]],
    *,
    opponents: Sequence[str],
) -> dict[str, Any]:
    opponent_set = set(opponents)
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "ok" or str(row.get("opponent_policy_id")) not in opponent_set:
            continue
        row_buckets = row.get("pair_index_buckets")
        if not isinstance(row_buckets, Mapping):
            continue
        for name, raw_bucket in row_buckets.items():
            if not isinstance(name, str) or not isinstance(raw_bucket, Mapping):
                continue
            bucket = buckets.setdefault(name, _empty_int_stats())
            bucket["shared_games"] += _int(raw_bucket.get("shared_games"))
            bucket["baseline_wins"] += _int(raw_bucket.get("baseline_wins"))
            bucket["candidate_wins"] += _int(raw_bucket.get("candidate_wins"))
            bucket["baseline_win_candidate_nonwin"] += _int(raw_bucket.get("baseline_win_candidate_nonwin"))
            bucket["baseline_nonwin_candidate_win"] += _int(raw_bucket.get("baseline_nonwin_candidate_win"))
            bucket["changed_outcome"] += _int(raw_bucket.get("changed_outcome"))
    return _finalize_pair_bucket_stats(buckets)


def _pair_index_bucket_name(pair_index: int, *, split: int) -> str:
    if split < 1:
        raise ValueError("pair_index_split must be >= 1")
    return f"pair_index_lt_{split}" if int(pair_index) < int(split) else f"pair_index_gte_{split}"


def _empty_int_stats() -> dict[str, int]:
    return {
        "shared_games": 0,
        "baseline_wins": 0,
        "candidate_wins": 0,
        "baseline_win_candidate_nonwin": 0,
        "baseline_nonwin_candidate_win": 0,
        "changed_outcome": 0,
    }


def _finalize_pair_bucket_stats(buckets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for name, raw_bucket in sorted(buckets.items()):
        shared_games = _int(raw_bucket.get("shared_games"))
        baseline_wins = _int(raw_bucket.get("baseline_wins"))
        candidate_wins = _int(raw_bucket.get("candidate_wins"))
        finalized[str(name)] = {
            "shared_games": shared_games,
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "delta_wins": candidate_wins - baseline_wins,
            "baseline_mean": None if shared_games <= 0 else baseline_wins / shared_games,
            "candidate_mean": None if shared_games <= 0 else candidate_wins / shared_games,
            "delta_mean": None if shared_games <= 0 else (candidate_wins - baseline_wins) / shared_games,
            "baseline_win_candidate_nonwin": _int(raw_bucket.get("baseline_win_candidate_nonwin")),
            "baseline_nonwin_candidate_win": _int(raw_bucket.get("baseline_nonwin_candidate_win")),
            "changed_outcome": _int(raw_bucket.get("changed_outcome")),
        }
    return finalized


def _load_summary_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _read_json_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"targeted-confirm summary missing rows: {path}")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        opponent = row.get("opponent_policy_id")
        if isinstance(opponent, str) and opponent:
            result[opponent] = row
    return result


def _load_row_records(row: Mapping[str, Any], *, summary_json: Path) -> tuple[EvalGameRecord, ...]:
    raw_summary_path = row.get("summary_path")
    if not isinstance(raw_summary_path, str) or not raw_summary_path:
        raise ValueError(f"summary row is missing summary_path in {summary_json}")
    matchup_summary_path = _resolve_path(raw_summary_path, base=summary_json.parent)
    episodes_path = matchup_summary_path.with_name("episodes.jsonl")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"episodes.jsonl not found next to matchup summary: {matchup_summary_path}")
    return load_eval_game_records(episodes_path)


def _resolve_path(raw_path: str, *, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    return base / path


def _record_key(record: EvalGameRecord) -> tuple[int, int, int]:
    return (int(record.pair_index), int(record.swap_index), int(record.episode_seed))


def _win_score(record: EvalGameRecord) -> int:
    return 1 if record.outcome == "W" else 0


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _flip_example(baseline: EvalGameRecord, candidate: EvalGameRecord) -> dict[str, Any]:
    return {
        "pair_index": int(baseline.pair_index),
        "swap_index": int(baseline.swap_index),
        "episode_seed": int(baseline.episode_seed),
        "baseline_outcome": baseline.outcome,
        "candidate_outcome": candidate.outcome,
        "baseline_decision_count": int(baseline.decision_count),
        "candidate_decision_count": int(candidate.decision_count),
        "baseline_pass_actions": int(baseline.pass_actions),
        "candidate_pass_actions": int(candidate.pass_actions),
        "baseline_pass_with_nonpass_available": int(baseline.pass_with_nonpass_available),
        "candidate_pass_with_nonpass_available": int(candidate.pass_with_nonpass_available),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
