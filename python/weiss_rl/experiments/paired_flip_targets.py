from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from weiss_rl.eval.harness import EvalGameRecord, write_episodes_jsonl
from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS
from weiss_rl.experiments.paired_outcome_compare import (
    _load_row_records,
    _load_summary_rows,
    _record_key,
    _resolve_opponent_row,
    _win_score,
)
from weiss_rl.experiments.paired_swing_report import _load_pool_tags, _opponent_tags

PairedFlipKind = Literal[
    "baseline_win_candidate_nonwin",
    "baseline_nonwin_candidate_win",
    "changed_outcome",
]
EpisodeSource = Literal["baseline", "candidate", "both"]

_SAFE_PATH_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PairedFlipTargetsConfig:
    baseline_summary_json: Path
    candidate_summary_json: Path
    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_opponents: tuple[str, ...] = ()
    opponent_pool_jsonls: tuple[Path, ...] = ()
    flip_kind: PairedFlipKind = "baseline_win_candidate_nonwin"
    pair_index_min: int | None = None
    pair_index_max: int | None = None
    max_targets_per_opponent: int | None = None
    episode_source: EpisodeSource = "candidate"
    episode_sets_dir: Path | None = None


def build_paired_flip_targets(config: PairedFlipTargetsConfig) -> dict[str, Any]:
    """Extract exact paired outcome flips and optional audit-ready episode subsets.

    Unlike paired-outcome examples, this function keeps every matching flip unless
    max_targets_per_opponent is set. Episode subsets are written as complete
    seat-swapped pairs so downstream replay audits can safely rerun them.
    """

    _validate_config(config)
    baseline_rows = _load_summary_rows(config.baseline_summary_json)
    candidate_rows = _load_summary_rows(config.candidate_summary_json)
    opponents = tuple(dict.fromkeys(config.opponents))
    pool_tags = _load_pool_tags(config.opponent_pool_jsonls)

    target_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    episode_sets: list[dict[str, Any]] = []
    for opponent in opponents:
        row_report, row_targets, row_episode_sets = _extract_opponent_targets(
            config=config,
            opponent=opponent,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            pool_tags=pool_tags,
        )
        target_rows.append(row_report)
        targets.extend(row_targets)
        episode_sets.extend(row_episode_sets)

    return {
        "kind": "paired_flip_targets_v1",
        "baseline": {
            "label": str(config.baseline_label),
            "summary_json": config.baseline_summary_json.as_posix(),
        },
        "candidate": {
            "label": str(config.candidate_label),
            "summary_json": config.candidate_summary_json.as_posix(),
        },
        "selection": {
            "opponents": list(opponents),
            "fixed_opponents": list(config.fixed_opponents),
            "learned_opponents": list(config.learned_opponents),
            "opponent_pool_jsonls": [path.as_posix() for path in config.opponent_pool_jsonls],
            "flip_kind": config.flip_kind,
            "pair_index_min": config.pair_index_min,
            "pair_index_max": config.pair_index_max,
            "max_targets_per_opponent": config.max_targets_per_opponent,
            "episode_source": config.episode_source,
            "episode_sets_dir": None if config.episode_sets_dir is None else config.episode_sets_dir.as_posix(),
        },
        "target_count": len(targets),
        "rows": target_rows,
        "targets": targets,
        "episode_sets": episode_sets,
        "seed_plan": _seed_plan(targets),
    }


def _extract_opponent_targets(
    *,
    config: PairedFlipTargetsConfig,
    opponent: str,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    pool_tags: Mapping[str, set[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_match = _resolve_opponent_row(baseline_rows, opponent)
    candidate_match = _resolve_opponent_row(candidate_rows, opponent)
    if baseline_match.row is None or candidate_match.row is None:
        return (
            {
                "opponent_policy_id": opponent,
                "status": "missing",
                "missing_baseline": baseline_match.row is None,
                "missing_candidate": candidate_match.row is None,
                "baseline_opponent_policy_id": baseline_match.opponent_policy_id,
                "candidate_opponent_policy_id": candidate_match.opponent_policy_id,
                "baseline_ambiguous_opponent_policy_ids": list(baseline_match.ambiguous_opponent_policy_ids),
                "candidate_ambiguous_opponent_policy_ids": list(candidate_match.ambiguous_opponent_policy_ids),
                "target_count": 0,
            },
            [],
            [],
        )

    baseline_records = _load_row_records(baseline_match.row, summary_json=config.baseline_summary_json)
    candidate_records = _load_row_records(candidate_match.row, summary_json=config.candidate_summary_json)
    tags = _opponent_tags(
        opponent,
        fixed_opponents=set(config.fixed_opponents),
        learned_opponents=set(config.learned_opponents),
        pool_tags=pool_tags,
    )
    baseline_by_key = {_record_key(record): record for record in baseline_records}
    candidate_by_key = {_record_key(record): record for record in candidate_records}
    shared_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    baseline_summary_path = _summary_path(baseline_match.row, summary_json=config.baseline_summary_json)
    candidate_summary_path = _summary_path(candidate_match.row, summary_json=config.candidate_summary_json)

    targets: list[dict[str, Any]] = []
    selected_pair_indices: set[int] = set()
    for key in shared_keys:
        baseline_record = baseline_by_key[key]
        candidate_record = candidate_by_key[key]
        if not _pair_index_allowed(int(baseline_record.pair_index), config=config):
            continue
        if not _matches_flip_kind(baseline_record, candidate_record, flip_kind=config.flip_kind):
            continue
        targets.append(
            _target_payload(
                opponent=opponent,
                baseline_opponent_policy_id=str(baseline_match.opponent_policy_id),
                candidate_opponent_policy_id=str(candidate_match.opponent_policy_id),
                tags=tags,
                baseline_matchup_summary_json=baseline_summary_path,
                candidate_matchup_summary_json=candidate_summary_path,
                baseline=baseline_record,
                candidate=candidate_record,
            )
        )
        selected_pair_indices.add(int(baseline_record.pair_index))
        if config.max_targets_per_opponent is not None and len(targets) >= int(config.max_targets_per_opponent):
            break

    episode_sets: list[dict[str, Any]] = []
    if config.episode_sets_dir is not None and selected_pair_indices:
        if config.episode_source in {"baseline", "both"}:
            episode_sets.append(
                _write_episode_set(
                    root=config.episode_sets_dir,
                    source="baseline",
                    opponent=opponent,
                    resolved_opponent=str(baseline_match.opponent_policy_id),
                    records=baseline_records,
                    pair_indices=selected_pair_indices,
                )
            )
        if config.episode_source in {"candidate", "both"}:
            episode_sets.append(
                _write_episode_set(
                    root=config.episode_sets_dir,
                    source="candidate",
                    opponent=opponent,
                    resolved_opponent=str(candidate_match.opponent_policy_id),
                    records=candidate_records,
                    pair_indices=selected_pair_indices,
                )
            )

    row_report = {
        "opponent_policy_id": opponent,
        "tags": sorted(tags),
        "baseline_opponent_policy_id": baseline_match.opponent_policy_id,
        "candidate_opponent_policy_id": candidate_match.opponent_policy_id,
        "status": "ok",
        "shared_games": len(shared_keys),
        "baseline_games": len(baseline_records),
        "candidate_games": len(candidate_records),
        "target_count": len(targets),
        "target_pair_count": len(selected_pair_indices),
        "target_pair_indices": sorted(selected_pair_indices),
        "episode_set_count": len(episode_sets),
        "baseline_wins": sum(_win_score(baseline_by_key[key]) for key in shared_keys),
        "candidate_wins": sum(_win_score(candidate_by_key[key]) for key in shared_keys),
    }
    row_report["delta_wins"] = int(row_report["candidate_wins"]) - int(row_report["baseline_wins"])
    return row_report, targets, episode_sets


def _validate_config(config: PairedFlipTargetsConfig) -> None:
    if config.flip_kind not in {
        "baseline_win_candidate_nonwin",
        "baseline_nonwin_candidate_win",
        "changed_outcome",
    }:
        raise ValueError(f"unsupported flip_kind: {config.flip_kind!r}")
    if config.episode_source not in {"baseline", "candidate", "both"}:
        raise ValueError(f"unsupported episode_source: {config.episode_source!r}")
    if config.pair_index_min is not None and int(config.pair_index_min) < 0:
        raise ValueError("pair_index_min must be >= 0")
    if config.pair_index_max is not None and int(config.pair_index_max) < 0:
        raise ValueError("pair_index_max must be >= 0")
    if (
        config.pair_index_min is not None
        and config.pair_index_max is not None
        and int(config.pair_index_min) > int(config.pair_index_max)
    ):
        raise ValueError("pair_index_min must be <= pair_index_max")
    if config.max_targets_per_opponent is not None and int(config.max_targets_per_opponent) <= 0:
        raise ValueError("max_targets_per_opponent must be positive when provided")
    if not config.opponents:
        raise ValueError("opponents must contain at least one opponent id")


def _pair_index_allowed(pair_index: int, *, config: PairedFlipTargetsConfig) -> bool:
    if config.pair_index_min is not None and int(pair_index) < int(config.pair_index_min):
        return False
    if config.pair_index_max is not None and int(pair_index) > int(config.pair_index_max):
        return False
    return True


def _matches_flip_kind(
    baseline: EvalGameRecord,
    candidate: EvalGameRecord,
    *,
    flip_kind: PairedFlipKind,
) -> bool:
    baseline_score = _win_score(baseline)
    candidate_score = _win_score(candidate)
    if flip_kind == "baseline_win_candidate_nonwin":
        return baseline_score == 1 and candidate_score == 0
    if flip_kind == "baseline_nonwin_candidate_win":
        return baseline_score == 0 and candidate_score == 1
    if flip_kind == "changed_outcome":
        return baseline.outcome != candidate.outcome
    raise ValueError(f"unsupported flip_kind: {flip_kind!r}")


def _target_payload(
    *,
    opponent: str,
    baseline_opponent_policy_id: str,
    candidate_opponent_policy_id: str,
    tags: set[str],
    baseline_matchup_summary_json: Path,
    candidate_matchup_summary_json: Path,
    baseline: EvalGameRecord,
    candidate: EvalGameRecord,
) -> dict[str, Any]:
    payload = {
        "opponent_policy_id": opponent,
        "baseline_opponent_policy_id": baseline_opponent_policy_id,
        "candidate_opponent_policy_id": candidate_opponent_policy_id,
        "tags": sorted(tags),
        "pair_index": int(baseline.pair_index),
        "swap_index": int(baseline.swap_index),
        "episode_seed": int(baseline.episode_seed),
        "baseline_matchup_summary_json": baseline_matchup_summary_json.as_posix(),
        "baseline_episodes_jsonl": baseline_matchup_summary_json.with_name("episodes.jsonl").as_posix(),
        "candidate_matchup_summary_json": candidate_matchup_summary_json.as_posix(),
        "candidate_episodes_jsonl": candidate_matchup_summary_json.with_name("episodes.jsonl").as_posix(),
        "baseline": _record_summary(baseline),
        "candidate": _record_summary(candidate),
        "decision_count_delta": int(candidate.decision_count) - int(baseline.decision_count),
        "pass_actions_delta": int(candidate.pass_actions) - int(baseline.pass_actions),
        "pass_with_nonpass_available_delta": int(candidate.pass_with_nonpass_available)
        - int(baseline.pass_with_nonpass_available),
    }
    payload["target_id"] = _target_id(payload)
    return payload


def _record_summary(record: EvalGameRecord) -> dict[str, Any]:
    return {
        "focal_policy_id": record.focal_policy_id,
        "opponent_policy_id": record.opponent_policy_id,
        "outcome": record.outcome,
        "focal_seat": int(record.focal_seat),
        "seat0_policy_id": record.seat0_policy_id,
        "seat1_policy_id": record.seat1_policy_id,
        "decision_count": int(record.decision_count),
        "pass_actions": int(record.pass_actions),
        "main_move_actions": int(record.main_move_actions),
        "pass_with_nonpass_available": int(record.pass_with_nonpass_available),
        "termination_reason": record.termination_reason,
        "engine_status": int(record.engine_status),
    }


def _write_episode_set(
    *,
    root: Path,
    source: Literal["baseline", "candidate"],
    opponent: str,
    resolved_opponent: str,
    records: Sequence[EvalGameRecord],
    pair_indices: set[int],
) -> dict[str, Any]:
    pair_index_set = {int(pair_index) for pair_index in pair_indices}
    selected = tuple(
        record
        for record in sorted(records, key=lambda item: (int(item.pair_index), int(item.swap_index), int(item.episode_seed)))
        if int(record.pair_index) in pair_index_set
    )
    _require_complete_seat_swaps(selected, source=source, opponent=opponent)
    out_dir = root / source / _path_slug(opponent)
    episodes_path = out_dir / "episodes.jsonl"
    write_episodes_jsonl(episodes_path, selected)
    paired_seeds = sorted({int(record.episode_seed) for record in selected})
    return {
        "source": source,
        "opponent_policy_id": opponent,
        "resolved_opponent_policy_id": resolved_opponent,
        "path": episodes_path.as_posix(),
        "record_count": len(selected),
        "pair_indices": sorted(pair_index_set),
        "paired_seeds": paired_seeds,
        "focal_policy_id": selected[0].focal_policy_id,
        "config_hash256": selected[0].config_hash256,
        "spec_hash256": selected[0].spec_hash256,
    }


def _require_complete_seat_swaps(
    records: Sequence[EvalGameRecord],
    *,
    source: str,
    opponent: str,
) -> None:
    if not records:
        raise ValueError(f"no records selected for {source}/{opponent}")
    by_pair: dict[int, set[int]] = {}
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    config_hashes = {record.config_hash256 for record in records}
    spec_hashes = {record.spec_hash256 for record in records}
    for record in records:
        by_pair.setdefault(int(record.pair_index), set()).add(int(record.swap_index))
    incomplete = {pair_index: swaps for pair_index, swaps in by_pair.items() if swaps != {0, 1}}
    if incomplete:
        raise ValueError(f"selected records must contain both swaps for {source}/{opponent}: {incomplete}")
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError(f"selected records must be one focal/opponent matchup for {source}/{opponent}")
    if len(config_hashes) != 1 or len(spec_hashes) != 1:
        raise ValueError(f"selected records must use one config/spec contract for {source}/{opponent}")


def _path_slug(value: str) -> str:
    stripped = _SAFE_PATH_CHARS_RE.sub("_", str(value).strip()).strip("._")
    return stripped[:160] if stripped else "opponent"


def _summary_path(row: Mapping[str, Any], *, summary_json: Path) -> Path:
    raw_summary_path = row.get("summary_path")
    if not isinstance(raw_summary_path, str) or not raw_summary_path:
        raise ValueError(f"summary row is missing summary_path in {summary_json}")
    path = Path(raw_summary_path)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    return summary_json.parent / path


def _target_id(payload: Mapping[str, Any]) -> str:
    stable = {
        "opponent_policy_id": payload.get("opponent_policy_id"),
        "baseline_opponent_policy_id": payload.get("baseline_opponent_policy_id"),
        "candidate_opponent_policy_id": payload.get("candidate_opponent_policy_id"),
        "pair_index": payload.get("pair_index"),
        "swap_index": payload.get("swap_index"),
        "episode_seed": payload.get("episode_seed"),
        "baseline_outcome": (payload.get("baseline") or {}).get("outcome")
        if isinstance(payload.get("baseline"), Mapping)
        else None,
        "candidate_outcome": (payload.get("candidate") or {}).get("outcome")
        if isinstance(payload.get("candidate"), Mapping)
        else None,
    }
    data = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(data).hexdigest()[:16]}"


def _seed_plan(targets: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[int]]]:
    buckets: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for target in targets:
        opponent = str(target.get("opponent_policy_id", ""))
        seed = int(target.get("episode_seed", 0))
        if not opponent or seed == 0:
            continue
        tags = set(str(tag) for tag in target.get("tags", []))
        group = _target_group(tags)
        buckets[group][opponent].add(seed)
    return {
        group: {opponent: sorted(seeds) for opponent, seeds in sorted(opponent_map.items())}
        for group, opponent_map in sorted(buckets.items())
    }


def _target_group(tags: set[str]) -> str:
    if "fixed" in tags:
        return "fixed"
    if "hard_negative" in tags:
        return "hard_negative"
    if "champion" in tags:
        return "champion"
    if "learned" in tags:
        return "learned"
    return "other"


def write_paired_flip_targets_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "EpisodeSource",
    "PairedFlipKind",
    "PairedFlipTargetsConfig",
    "build_paired_flip_targets",
    "write_paired_flip_targets_json",
]
