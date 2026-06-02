"""Final-eval summary access helpers for paper-readiness checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.eval.readiness.fields import (
    as_int,
    load_json_object,
    mapping,
)


def policy_ids(payload: Mapping[str, Any]) -> list[str]:
    raw_policy_ids = payload.get("policy_ids")
    if not isinstance(raw_policy_ids, list) or any(not isinstance(item, str) for item in raw_policy_ids):
        raise ValueError("final_eval summary must include string policy_ids")
    return list(raw_policy_ids)


def matchups(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include matchups")
    matchup_payloads: list[Mapping[str, Any]] = []
    for index, matchup in enumerate(raw_matchups):
        matchup_payloads.append(mapping(matchup, context=f"matchups[{index}]"))
    return matchup_payloads


def canonical_unordered_matchups(
    matchups: Sequence[Mapping[str, Any]],
    *,
    policy_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    selected: dict[tuple[int, int], tuple[int, Mapping[str, Any]]] = {}
    for index, matchup in enumerate(matchups):
        focal_index = matchup_policy_index(
            matchup,
            index_field="focal_policy_index",
            policy_field="focal_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        opponent_index = matchup_policy_index(
            matchup,
            index_field="opponent_policy_index",
            policy_field="opponent_policy_id",
            policy_ids=policy_ids,
            context=f"matchups[{index}]",
        )
        key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
        rank = 0 if focal_index <= opponent_index else 1
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, matchup)
    return [selected[key][1] for key in sorted(selected)]


def load_matchup_diagnostics(
    *,
    final_eval_dir: Path,
    matchups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for matchup in matchups:
        diagnostics_path = final_eval_dir / str(matchup["diagnostics_path"])
        diagnostics = load_json_object(diagnostics_path)
        seat_results = mapping(diagnostics.get("seat_results"), context=f"{diagnostics_path}:seat_results")
        seat0_wins = as_int(seat_results.get("seat0_wins"), context=f"{diagnostics_path}:seat0_wins")
        seat1_wins = as_int(seat_results.get("seat1_wins"), context=f"{diagnostics_path}:seat1_wins")
        draws = as_int(seat_results.get("draws"), context=f"{diagnostics_path}:draws")
        truncations = as_int(seat_results.get("truncations"), context=f"{diagnostics_path}:truncations")
        engine_errors = as_int(seat_results.get("engine_errors"), context=f"{diagnostics_path}:engine_errors")
        decisive_games = seat0_wins + seat1_wins
        payloads.append(
            {
                "focal_policy_id": str(matchup["focal_policy_id"]),
                "opponent_policy_id": str(matchup["opponent_policy_id"]),
                "diagnostics_path": str(matchup["diagnostics_path"]),
                "seat0_wins": seat0_wins,
                "seat1_wins": seat1_wins,
                "draws": draws,
                "truncations": truncations,
                "engine_errors": engine_errors,
                "decisive_games": decisive_games,
                "total_games": decisive_games + draws + truncations,
            }
        )
    return payloads


def matchup_policy_index(
    matchup: Mapping[str, Any],
    *,
    index_field: str,
    policy_field: str,
    policy_ids: Sequence[str],
    context: str,
) -> int:
    raw_index = matchup.get(index_field)
    if raw_index is not None:
        index = as_int(raw_index, context=f"{context}.{index_field}")
        if index < 0 or index >= len(policy_ids):
            raise ValueError(
                f"{context}.{index_field}={index} is out of range for policy_ids with length {len(policy_ids)}"
            )
        return index
    policy_id = matchup.get(policy_field)
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError(f"{context}.{policy_field} must be a non-empty string")
    try:
        return policy_ids.index(policy_id)
    except ValueError as exc:
        raise ValueError(f"{context}.{policy_field}={policy_id!r} is missing from policy_ids") from exc


def metadata_focal_policy_id(payload: Mapping[str, Any]) -> str | None:
    metadata = mapping(payload.get("metadata", {}), context="metadata")
    for path in (
        ("focal_policy_id",),
        ("recommended_focal_policy_id",),
        ("focal_policy", "policy_id"),
        ("selection", "focal_policy_id"),
    ):
        value = nested_optional_string(metadata, path=path)
        if value is not None:
            return value
    return None


def nested_optional_string(payload: Mapping[str, Any], *, path: Sequence[str]) -> str | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, str):
        normalized = current.strip()
        return normalized or None
    return None


def matrix_cell(payload: Mapping[str, Any], *, field: str, row: int, column: int) -> Any:
    payload_matrix = matrix(payload, field=field)
    try:
        matrix_row = payload_matrix[row]
        if not isinstance(matrix_row, list):
            raise TypeError
        return matrix_row[column]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"matrix {field!r} is missing cell [{row}][{column}]") from exc


def matrix(payload: Mapping[str, Any], *, field: str) -> list[Any]:
    matrices = mapping(payload.get("matrices"), context="matrices")
    matrix_payload = mapping(matrices.get(field), context=f"matrices.{field}")
    values = matrix_payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"matrices.{field}.values must be a list")
    return values


def posterior_samples(payload: Mapping[str, Any], *, focal_index: int, opponent_index: int) -> list[float]:
    posterior_payload = mapping(payload.get("posterior_samples"), context="posterior_samples")
    values = posterior_payload.get("values")
    if not isinstance(values, list):
        raise ValueError("posterior_samples.values must be a list")
    try:
        row = values[focal_index]
        if not isinstance(row, list):
            raise TypeError
        samples = row[opponent_index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"posterior_samples.values is missing cell [{focal_index}][{opponent_index}]") from exc
    if not isinstance(samples, list):
        raise ValueError("posterior sample cell must be a list")
    return [float(sample) for sample in samples]
