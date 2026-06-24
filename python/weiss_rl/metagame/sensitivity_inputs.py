"""Input loading and integrity checks for metagame sensitivity reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.eval.analysis.export import load_eval_game_records
from weiss_rl.eval.simulator.harness import EvalGameRecord


@dataclass(frozen=True, slots=True)
class FinalEvalMatchup:
    focal_policy_id: str
    opponent_policy_id: str
    focal_policy_index: int
    opponent_policy_index: int
    episodes_path: Path
    records: tuple[EvalGameRecord, ...]


@dataclass(frozen=True, slots=True)
class FinalEvalContext:
    policy_ids: tuple[str, ...]
    matchups: tuple[FinalEvalMatchup, ...]


def load_final_eval_context(final_eval_dir: Path) -> FinalEvalContext:
    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_ids = tuple(_require_str_list(payload.get("policy_ids"), field_name="policy_ids"))
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("final_eval policy_ids must be unique")
    policy_index = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include a matchups list")

    matchups: list[FinalEvalMatchup] = []
    canonical_keys: set[tuple[int, int]] = set()
    for item in raw_matchups:
        if not isinstance(item, dict):
            raise ValueError("final_eval matchups must contain objects")
        focal_policy_id = _require_matchup_policy_id(
            item,
            field_name="focal_policy_id",
            policy_index=policy_index,
        )
        opponent_policy_id = _require_matchup_policy_id(
            item,
            field_name="opponent_policy_id",
            policy_index=policy_index,
        )
        focal_index = _require_matchup_policy_index(
            item,
            index_field="focal_policy_index",
            policy_id=focal_policy_id,
            policy_index=policy_index,
        )
        opponent_index = _require_matchup_policy_index(
            item,
            index_field="opponent_policy_index",
            policy_id=opponent_policy_id,
            policy_index=policy_index,
        )
        if focal_index > opponent_index:
            raise ValueError("final_eval matchups must be canonical with focal_policy_index <= opponent_policy_index")
        canonical_key = (focal_index, opponent_index)
        if canonical_key in canonical_keys:
            raise ValueError(
                "final_eval summary contains duplicate canonical matchup: "
                f"({focal_policy_id!r}, {opponent_policy_id!r})"
            )
        canonical_keys.add(canonical_key)
        episodes_path = _resolve_final_eval_episodes_path(
            final_eval_dir=final_eval_dir,
            value=item.get("episodes_path"),
            expected_relative_path=_expected_final_eval_episodes_path(
                focal_policy_index=focal_index,
                opponent_policy_index=opponent_index,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
            ),
        )
        records = load_eval_game_records(episodes_path)
        _validate_matchup_records(
            records=records,
            episodes_path=episodes_path,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
        )
        matchups.append(
            FinalEvalMatchup(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                focal_policy_index=focal_index,
                opponent_policy_index=opponent_index,
                episodes_path=episodes_path,
                records=records,
            )
        )
    expected_matchups = (len(policy_ids) * (len(policy_ids) + 1)) // 2
    if len(canonical_keys) != expected_matchups:
        raise ValueError(
            "final_eval summary is missing canonical matchups: "
            f"expected {expected_matchups}, found {len(canonical_keys)}"
        )
    return FinalEvalContext(policy_ids=policy_ids, matchups=tuple(matchups))


def observed_pair_count(records: Sequence[EvalGameRecord]) -> int:
    return len({record.pair_index for record in records})


def _require_matchup_policy_id(
    item: Mapping[str, Any],
    *,
    field_name: str,
    policy_index: Mapping[str, int],
) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"final_eval matchup {field_name} must be a non-empty string")
    if value not in policy_index:
        raise ValueError(f"final_eval matchup {field_name}={value!r} is missing from policy_ids")
    return value


def _require_matchup_policy_index(
    item: Mapping[str, Any],
    *,
    index_field: str,
    policy_id: str,
    policy_index: Mapping[str, int],
) -> int:
    if index_field not in item:
        raise ValueError(f"final_eval matchup missing {index_field}")
    raw_value = item[index_field]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"final_eval matchup {index_field} must be an integer")
    index = int(raw_value)
    expected_index = policy_index[policy_id]
    if index != expected_index:
        raise ValueError(
            f"final_eval matchup {index_field}={index} does not match policy_ids position "
            f"for {policy_id!r} (expected {expected_index})"
        )
    return index


def _expected_final_eval_episodes_path(
    *,
    focal_policy_index: int,
    opponent_policy_index: int,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> Path:
    matchup_dir = (
        f"{focal_policy_index:02d}_{_slug(focal_policy_id)}__vs__"
        f"{opponent_policy_index:02d}_{_slug(opponent_policy_id)}"
    )
    return Path("matchups") / matchup_dir / "episodes.jsonl"


def _slug(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    slug = "_".join(part for part in parts if part)
    return slug or "policy"


def _resolve_final_eval_episodes_path(*, final_eval_dir: Path, value: Any, expected_relative_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("final_eval matchup episodes_path must be a non-empty string")
    raw_path = Path(value)
    if raw_path.is_absolute() or value.startswith(("/", "\\")):
        raise ValueError(
            f"final_eval matchup episodes_path must be relative to the final_eval root, got absolute path: {value!r}"
        )
    resolved_root = final_eval_dir.resolve()
    resolved_path = (final_eval_dir / raw_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"final_eval matchup episodes_path resolves outside the final_eval root: {value!r}") from exc
    expected = expected_relative_path.as_posix()
    if value != expected:
        raise ValueError(
            "final_eval matchup episodes_path must equal canonical final_eval artifact path: "
            f"expected {expected!r}, got {value!r}"
        )
    return resolved_path


def _validate_matchup_records(
    *,
    records: Sequence[EvalGameRecord],
    episodes_path: Path,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> None:
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError(
            "final_eval matchup episodes must contain exactly one focal_policy_id and "
            f"one opponent_policy_id: {episodes_path}"
        )
    actual_focal_policy_id = next(iter(focal_ids))
    actual_opponent_policy_id = next(iter(opponent_ids))
    if actual_focal_policy_id != focal_policy_id or actual_opponent_policy_id != opponent_policy_id:
        raise ValueError(
            "final_eval matchup episodes do not match summary metadata for "
            f"{episodes_path}: summary expects ({focal_policy_id!r}, {opponent_policy_id!r}), "
            f"loaded ({actual_focal_policy_id!r}, {actual_opponent_policy_id!r})"
        )


def _require_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return [str(item) for item in value]


__all__ = ["FinalEvalContext", "FinalEvalMatchup", "load_final_eval_context", "observed_pair_count"]
