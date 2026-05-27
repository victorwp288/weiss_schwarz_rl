"""Summarize diagnostic league progress from training scalar logs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUTCOME_RE = re.compile(
    r"^collector_outcome_vs_(?P<opponent>.+)_(?P<field>wins|losses|draws|games|win_rate|decisive_win_rate|timeouts)$"
)
POLICY_EXPOSURE_GROUPS = (
    "sampled",
    "heuristic_public",
    "heuristic_public_variant",
    "noleague_baseline",
    "champion",
    "recent",
    "hard_negative",
    "warmup_snapshot",
)
POLICY_EXPOSURE_RE = re.compile(
    r"^collector_pfsp_(?P<group>"
    + "|".join(re.escape(group) for group in POLICY_EXPOSURE_GROUPS)
    + r")_policy_envs__(?P<policy>.+)$"
)

FIXED_OPPONENT_SLUGS = {
    "b0_randomlegal",
    "b1_noleague_baseline",
    "b2_heuristicpublic",
    "b3_heuristicpublicaggro",
    "b4_heuristicpubliccontrol",
}

EXPOSURE_KEYS = (
    "pfsp_champion_envs",
    "pfsp_hard_negative_envs",
    "pfsp_heuristic_public_envs",
    "pfsp_heuristic_public_variant_envs",
    "pfsp_noleague_baseline_envs",
    "pfsp_mirror_envs",
    "pfsp_recent_envs",
    "pfsp_warmup_snapshot_envs",
    "pfsp_sampled_envs",
)

POOL_SIZE_KEYS = (
    "pfsp_pool_size",
    "pfsp_champion_pool_size",
    "pfsp_hard_negative_pool_size",
    "pfsp_recent_pool_size",
    "pfsp_quarantined_opponents",
)


@dataclass(frozen=True)
class OutcomeRow:
    update_count: int
    opponent_slug: str
    wins: int
    losses: int
    draws: int
    games: int
    timeouts: int
    win_rate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "opponent_slug": self.opponent_slug,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "timeouts": self.timeouts,
            "win_rate": self.win_rate,
        }


def build_league_progress_summary(
    *,
    scalars_jsonl: Path,
    learned_opponents: Iterable[str] = (),
    hard_negative_opponents: Iterable[str] = (),
    notes: str = "",
) -> dict[str, Any]:
    """Build a diagnostic summary from training/logs/scalars.jsonl.

    The scalar log contains unpaired, on-policy collector outcomes. This helper
    intentionally labels the result as diagnostic so it cannot be mistaken for
    thesis-grade paired confirmation evidence.
    """

    learned_slugs = {slugify_opponent(item) for item in learned_opponents if str(item).strip()}
    hard_negative_slugs = {slugify_opponent(item) for item in hard_negative_opponents if str(item).strip()}
    rows_by_opponent: dict[str, list[OutcomeRow]] = defaultdict(list)
    exposure_totals = {key: 0.0 for key in EXPOSURE_KEYS}
    exposure_max = {key: 0.0 for key in EXPOSURE_KEYS}
    pool_size_max = {key: 0.0 for key in POOL_SIZE_KEYS}
    pool_size_last = {key: 0.0 for key in POOL_SIZE_KEYS}
    policy_exposure_totals: dict[str, dict[str, float]] = {
        group: defaultdict(float) for group in POLICY_EXPOSURE_GROUPS
    }
    policy_exposure_max: dict[str, dict[str, float]] = {
        group: defaultdict(float) for group in POLICY_EXPOSURE_GROUPS
    }
    update_count_min: int | None = None
    update_count_max: int | None = None
    records_seen = 0

    for record in _iter_jsonl(scalars_jsonl):
        records_seen += 1
        update_count = int(float(record.get("update_count", records_seen)))
        update_count_min = update_count if update_count_min is None else min(update_count_min, update_count)
        update_count_max = update_count if update_count_max is None else max(update_count_max, update_count)

        for key in EXPOSURE_KEYS:
            value = _float_or_zero(record.get(key))
            exposure_totals[key] += value
            exposure_max[key] = max(exposure_max[key], value)
        for key in POOL_SIZE_KEYS:
            value = _float_or_zero(record.get(key))
            pool_size_last[key] = value
            pool_size_max[key] = max(pool_size_max[key], value)

        grouped_fields: dict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in record.items():
            exposure_match = POLICY_EXPOSURE_RE.match(str(key))
            if exposure_match is not None:
                group = exposure_match.group("group")
                policy_slug = exposure_match.group("policy")
                exposure_value = _float_or_zero(value)
                policy_exposure_totals[group][policy_slug] += exposure_value
                policy_exposure_max[group][policy_slug] = max(policy_exposure_max[group][policy_slug], exposure_value)
                continue
            match = OUTCOME_RE.match(str(key))
            if match is None:
                continue
            grouped_fields[match.group("opponent")][match.group("field")] = value
        for opponent_slug, fields in grouped_fields.items():
            games = int(_float_or_zero(fields.get("games")))
            if games <= 0:
                continue
            wins = int(_float_or_zero(fields.get("wins")))
            losses = int(_float_or_zero(fields.get("losses")))
            draws = int(_float_or_zero(fields.get("draws")))
            timeouts = int(_float_or_zero(fields.get("timeouts")))
            rows_by_opponent[opponent_slug].append(
                OutcomeRow(
                    update_count=update_count,
                    opponent_slug=opponent_slug,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                    games=games,
                    timeouts=timeouts,
                    win_rate=float(fields.get("win_rate", wins / games)),
                )
            )

    opponent_summaries = [
        _summarize_opponent(
            opponent_slug=opponent_slug,
            rows=rows,
            learned_slugs=learned_slugs,
            hard_negative_slugs=hard_negative_slugs,
        )
        for opponent_slug, rows in sorted(rows_by_opponent.items())
    ]
    group_summaries = _summarize_groups(opponent_summaries)
    return {
        "kind": "league_progress_summary_v1",
        "evidence_grade": "diagnostic_unpaired_training_collector",
        "notes": notes,
        "scalars_jsonl": scalars_jsonl.as_posix(),
        "records_seen": records_seen,
        "update_count_min": update_count_min,
        "update_count_max": update_count_max,
        "exposure_totals": exposure_totals,
        "exposure_max": exposure_max,
        "policy_exposure_totals": _sorted_nested_float_maps(policy_exposure_totals),
        "policy_exposure_max": _sorted_nested_float_maps(policy_exposure_max),
        "pool_size_last": pool_size_last,
        "pool_size_max": pool_size_max,
        "learned_opponents": sorted(learned_slugs),
        "hard_negative_opponents": sorted(hard_negative_slugs),
        "opponents": opponent_summaries,
        "groups": group_summaries,
    }


def slugify_opponent(opponent: str) -> str:
    normalized = str(opponent).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def write_league_progress_summary(summary: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"expected object in {path} at line {line_no}")
            yield payload


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sorted_nested_float_maps(values: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {
        group: {policy_slug: float(amount) for policy_slug, amount in sorted(group_values.items()) if amount != 0.0}
        for group, group_values in values.items()
    }


def _classify_opponent(
    *,
    opponent_slug: str,
    learned_slugs: set[str],
    hard_negative_slugs: set[str],
) -> str:
    if _slug_matches_any(opponent_slug, hard_negative_slugs):
        return "hard_negative"
    if opponent_slug in FIXED_OPPONENT_SLUGS:
        return "fixed_baseline"
    if _slug_matches_any(opponent_slug, learned_slugs) or opponent_slug.startswith("seed_"):
        return "imported_learned"
    if opponent_slug.startswith("policy_"):
        return "self_league_snapshot"
    return "other"


def _slug_matches_any(opponent_slug: str, requested_slugs: set[str]) -> bool:
    return any(_is_seed_wrapped_suffix_match(opponent_slug, requested_slug) for requested_slug in requested_slugs)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _summarize_opponent(
    *,
    opponent_slug: str,
    rows: list[OutcomeRow],
    learned_slugs: set[str],
    hard_negative_slugs: set[str],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row.update_count)
    first = ordered[0]
    last = ordered[-1]
    wins = sum(row.wins for row in ordered)
    games = sum(row.games for row in ordered)
    losses = sum(row.losses for row in ordered)
    draws = sum(row.draws for row in ordered)
    best = max(ordered, key=lambda row: (row.win_rate, row.games, row.update_count))
    worst = min(ordered, key=lambda row: (row.win_rate, -row.games, row.update_count))
    return {
        "opponent_slug": opponent_slug,
        "group": _classify_opponent(
            opponent_slug=opponent_slug,
            learned_slugs=learned_slugs,
            hard_negative_slugs=hard_negative_slugs,
        ),
        "updates_observed": len(ordered),
        "first": first.to_json(),
        "last": last.to_json(),
        "best": best.to_json(),
        "worst": worst.to_json(),
        "delta_last_minus_first": last.win_rate - first.win_rate,
        "total": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games,
            "win_rate": wins / games if games else None,
        },
    }


def _summarize_groups(opponent_summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in opponent_summaries:
        grouped[str(summary["group"])].append(summary)

    output: dict[str, dict[str, Any]] = {}
    for group, summaries in sorted(grouped.items()):
        total_wins = sum(int(item["total"]["wins"]) for item in summaries)
        total_games = sum(int(item["total"]["games"]) for item in summaries)
        last_wins = sum(int(item["last"]["wins"]) for item in summaries)
        last_games = sum(int(item["last"]["games"]) for item in summaries)
        first_wins = sum(int(item["first"]["wins"]) for item in summaries)
        first_games = sum(int(item["first"]["games"]) for item in summaries)
        output[group] = {
            "opponent_count": len(summaries),
            "total": {
                "wins": total_wins,
                "games": total_games,
                "win_rate": total_wins / total_games if total_games else None,
            },
            "first_observed": {
                "wins": first_wins,
                "games": first_games,
                "win_rate": first_wins / first_games if first_games else None,
            },
            "last_observed": {
                "wins": last_wins,
                "games": last_games,
                "win_rate": last_wins / last_games if last_games else None,
            },
        }
        first_rate = output[group]["first_observed"]["win_rate"]
        last_rate = output[group]["last_observed"]["win_rate"]
        output[group]["delta_last_minus_first"] = None if first_rate is None or last_rate is None else last_rate - first_rate
    return output


__all__ = [
    "build_league_progress_summary",
    "slugify_opponent",
    "write_league_progress_summary",
]
