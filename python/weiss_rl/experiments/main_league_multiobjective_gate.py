from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXED_THESIS_OPPONENTS = (
    "B0 RandomLegal",
    "B1 NoLeague baseline",
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)


@dataclass(frozen=True, slots=True)
class MultiObjectiveGateConfig:
    candidate_summary_jsons: tuple[Path, ...]
    reference_summary_jsons: tuple[Path, ...] = ()
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_opponents: tuple[str, ...] = ()
    opponent_aliases: Mapping[str, str] | None = None
    min_fixed_score: float = 0.5
    max_fixed_reference_drop: float = 0.0
    min_learned_score: float = 0.5
    min_learned_mean: float = 0.5
    min_learned_reference_delta: float | None = 0.0
    max_learned_reference_drop: float | None = None


def evaluate_main_league_multiobjective_gate(config: MultiObjectiveGateConfig) -> dict[str, Any]:
    candidate_scores = _load_score_table(config.candidate_summary_jsons)
    reference_scores = _load_score_table(config.reference_summary_jsons, aliases=config.opponent_aliases or {})
    fixed = _evaluate_rows(
        group_name="fixed_baselines",
        opponents=config.fixed_opponents,
        candidate_scores=candidate_scores,
        reference_scores=reference_scores,
        min_score=float(config.min_fixed_score),
        max_reference_drop=float(config.max_fixed_reference_drop),
    )
    learned = _evaluate_rows(
        group_name="learned_opponents",
        opponents=config.learned_opponents,
        candidate_scores=candidate_scores,
        reference_scores=reference_scores,
        min_score=float(config.min_learned_score),
        max_reference_drop=config.max_learned_reference_drop,
    )
    learned_reference_delta = _group_reference_delta(learned)
    group_failures: list[dict[str, Any]] = []
    learned_mean = learned["mean"]
    if config.learned_opponents and (learned_mean is None or float(learned_mean) < float(config.min_learned_mean)):
        group_failures.append(
            {
                "group": "learned_opponents",
                "reason": "below_min_group_mean",
                "mean": learned_mean,
                "threshold": float(config.min_learned_mean),
            }
        )
    if (
        config.learned_opponents
        and config.min_learned_reference_delta is not None
        and learned_reference_delta is not None
        and learned_reference_delta < float(config.min_learned_reference_delta)
    ):
        group_failures.append(
            {
                "group": "learned_opponents",
                "reason": "below_min_group_reference_delta",
                "delta": learned_reference_delta,
                "threshold": float(config.min_learned_reference_delta),
            }
        )

    failures = [*fixed["failures"], *learned["failures"], *group_failures]
    return {
        "kind": "main_league_multiobjective_gate_v1",
        "passed": not failures,
        "failures": failures,
        "candidate_summary_jsons": [path.as_posix() for path in config.candidate_summary_jsons],
        "reference_summary_jsons": [path.as_posix() for path in config.reference_summary_jsons],
        "opponent_aliases": dict(config.opponent_aliases or {}),
        "thresholds": {
            "min_fixed_score": float(config.min_fixed_score),
            "max_fixed_reference_drop": float(config.max_fixed_reference_drop),
            "min_learned_score": float(config.min_learned_score),
            "min_learned_mean": float(config.min_learned_mean),
            "min_learned_reference_delta": config.min_learned_reference_delta,
            "max_learned_reference_drop": config.max_learned_reference_drop,
        },
        "groups": {
            "fixed_baselines": fixed,
            "learned_opponents": {
                **learned,
                "reference_delta": learned_reference_delta,
            },
        },
    }


def _evaluate_rows(
    *,
    group_name: str,
    opponents: Sequence[str],
    candidate_scores: Mapping[str, dict[str, Any]],
    reference_scores: Mapping[str, dict[str, Any]],
    min_score: float,
    max_reference_drop: float | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    wins = 0
    games = 0
    present_count = 0
    for opponent in opponents:
        candidate_match = _resolve_score_row(candidate_scores, opponent)
        reference_match = _resolve_score_row(reference_scores, opponent)
        candidate = candidate_match.row
        reference = reference_match.row
        candidate_mean = _optional_float(None if candidate is None else candidate.get("mean"))
        reference_mean = _optional_float(None if reference is None else reference.get("mean"))
        delta = None if candidate_mean is None or reference_mean is None else candidate_mean - reference_mean
        row = {
            "opponent_policy_id": opponent,
            "candidate_opponent_policy_id": candidate_match.opponent_policy_id,
            "reference_opponent_policy_id": reference_match.opponent_policy_id,
            "candidate_ambiguous_opponent_policy_ids": list(candidate_match.ambiguous_opponent_policy_ids),
            "reference_ambiguous_opponent_policy_ids": list(reference_match.ambiguous_opponent_policy_ids),
            "mean": candidate_mean,
            "reference": reference_mean,
            "delta": delta,
            "wins": None if candidate is None else candidate.get("wins"),
            "games": None if candidate is None else candidate.get("games"),
            "summary_path": None if candidate is None else candidate.get("summary_path"),
        }
        rows.append(row)
        if candidate_match.ambiguous_opponent_policy_ids:
            failures.append(
                {
                    "group": group_name,
                    "opponent": opponent,
                    "reason": "ambiguous_candidate_score",
                    "matches": list(candidate_match.ambiguous_opponent_policy_ids),
                }
            )
            continue
        if reference_match.ambiguous_opponent_policy_ids:
            failures.append(
                {
                    "group": group_name,
                    "opponent": opponent,
                    "reason": "ambiguous_reference_score",
                    "matches": list(reference_match.ambiguous_opponent_policy_ids),
                }
            )
            continue
        if candidate is None or candidate_mean is None:
            failures.append({"group": group_name, "opponent": opponent, "reason": "missing_candidate_score"})
            continue
        present_count += 1
        row_games = _optional_int(candidate.get("games"))
        row_wins = _optional_int(candidate.get("wins"))
        if row_games is not None and row_wins is not None:
            wins += row_wins
            games += row_games
        if candidate_mean < min_score:
            failures.append(
                {
                    "group": group_name,
                    "opponent": opponent,
                    "reason": "below_min_score",
                    "score": candidate_mean,
                    "threshold": min_score,
                }
            )
        if max_reference_drop is not None and delta is not None and delta < -float(max_reference_drop):
            failures.append(
                {
                    "group": group_name,
                    "opponent": opponent,
                    "reason": "below_reference_drop_limit",
                    "score": candidate_mean,
                    "reference": reference_mean,
                    "delta": delta,
                    "threshold": -float(max_reference_drop),
                }
            )
    mean = None if games <= 0 else wins / games
    return {
        "opponents": list(opponents),
        "rows": rows,
        "wins": wins,
        "games": games,
        "mean": mean,
        "present_count": present_count,
        "expected_count": len(tuple(opponents)),
        "failures": failures,
    }


@dataclass(frozen=True, slots=True)
class _ResolvedScoreRow:
    row: Mapping[str, Any] | None
    opponent_policy_id: str | None
    ambiguous_opponent_policy_ids: tuple[str, ...] = ()


def _resolve_score_row(scores: Mapping[str, dict[str, Any]], opponent: str) -> _ResolvedScoreRow:
    row = scores.get(opponent)
    if row is not None:
        return _ResolvedScoreRow(row=row, opponent_policy_id=opponent)
    suffix_matches = tuple(
        row_opponent
        for row_opponent in scores
        if _is_seed_wrapped_suffix_match(str(row_opponent), str(opponent))
    )
    if len(suffix_matches) == 1:
        resolved = suffix_matches[0]
        return _ResolvedScoreRow(row=scores[resolved], opponent_policy_id=resolved)
    if len(suffix_matches) > 1:
        return _ResolvedScoreRow(
            row=None,
            opponent_policy_id=None,
            ambiguous_opponent_policy_ids=tuple(sorted(suffix_matches)),
        )
    return _ResolvedScoreRow(row=None, opponent_policy_id=None)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _group_reference_delta(group: Mapping[str, Any]) -> float | None:
    rows = group.get("rows")
    if not isinstance(rows, list):
        return None
    candidate_values: list[float] = []
    reference_values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate = _optional_float(row.get("mean"))
        reference = _optional_float(row.get("reference"))
        if candidate is None or reference is None:
            continue
        candidate_values.append(candidate)
        reference_values.append(reference)
    if not candidate_values:
        return None
    return (sum(candidate_values) / len(candidate_values)) - (sum(reference_values) / len(reference_values))


def _load_score_table(paths: Sequence[Path], *, aliases: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _read_json_object(path)
        anchor_scores = payload.get("anchor_scores")
        if isinstance(anchor_scores, Mapping):
            for opponent, raw_mean in anchor_scores.items():
                if not isinstance(opponent, str) or not opponent:
                    continue
                mean = _optional_float(raw_mean)
                if mean is None:
                    continue
                mapped_opponent = str((aliases or {}).get(opponent, opponent))
                table[mapped_opponent] = {
                    "opponent_policy_id": mapped_opponent,
                    "source_opponent_policy_id": opponent,
                    "mean": mean,
                    "wins": None,
                    "games": None,
                    "summary_path": path.as_posix(),
                    "source_summary_path": path.as_posix(),
                }
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"targeted-confirm summary missing rows: {path}")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = row.get("opponent_policy_id")
            if not isinstance(opponent, str) or not opponent:
                continue
            mapped_opponent = str((aliases or {}).get(opponent, opponent))
            mean = _optional_float(row.get("mean"))
            if mean is None:
                continue
            table[mapped_opponent] = {
                "opponent_policy_id": mapped_opponent,
                "source_opponent_policy_id": opponent,
                "mean": mean,
                "wins": _optional_int(row.get("wins")),
                "games": _optional_int(row.get("games")),
                "summary_path": str(row.get("summary_path") or path.as_posix()),
                "source_summary_path": path.as_posix(),
            }
    return table


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
