"""Matchup coverage and artifact-reference checks for final-eval readiness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.eval.readiness.fields import (
    require_relative_artifact_path as _require_relative_artifact_path,
)
from weiss_rl.eval.readiness.final_eval_summary import (
    matchup_policy_index as _matchup_policy_index,
)
from weiss_rl.eval.readiness.final_eval_summary import (
    matchups as _matchups,
)


def build_final_eval_matchup_contract(
    *,
    final_eval_dir: Path,
    summary_payload: dict[str, Any],
    policy_ids: Sequence[str],
) -> dict[str, Any]:
    expected_keys = {(left, right) for left in range(len(policy_ids)) for right in range(left, len(policy_ids))}
    observed_keys: dict[tuple[int, int], str] = {}
    duplicate_matchups: list[str] = []
    noncanonical_matchups: list[str] = []
    reference_failures: list[str] = []

    try:
        for index, matchup in enumerate(_matchups(summary_payload)):
            focal_index = _matchup_policy_index(
                matchup,
                index_field="focal_policy_index",
                policy_field="focal_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            opponent_index = _matchup_policy_index(
                matchup,
                index_field="opponent_policy_index",
                policy_field="opponent_policy_id",
                policy_ids=policy_ids,
                context=f"matchups[{index}]",
            )
            pair_label = f"{policy_ids[focal_index]}__vs__{policy_ids[opponent_index]}"
            if focal_index > opponent_index:
                noncanonical_matchups.append(pair_label)
            key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
            if key in observed_keys:
                duplicate_matchups.append(pair_label)
            else:
                observed_keys[key] = pair_label

            _validate_matchup_artifact_references(
                final_eval_dir=final_eval_dir,
                matchup=matchup,
                matchup_index=index,
                reference_failures=reference_failures,
            )
    except ValueError as exc:
        return {
            "passed": False,
            "expected_matchup_count": len(expected_keys),
            "observed_matchup_count": len(observed_keys),
            "missing_matchups": [],
            "duplicate_matchups": duplicate_matchups,
            "noncanonical_matchups": noncanonical_matchups,
            "reference_failures": [str(exc)],
            "reason": "invalid_matchup_index",
            "message": str(exc),
        }

    missing_matchups = [
        f"{policy_ids[left]}__vs__{policy_ids[right]}" for left, right in sorted(expected_keys - set(observed_keys))
    ]
    passed = not duplicate_matchups and not noncanonical_matchups and not missing_matchups and not reference_failures

    return {
        "passed": passed,
        "expected_matchup_count": len(expected_keys),
        "observed_matchup_count": len(observed_keys),
        "missing_matchups": missing_matchups,
        "duplicate_matchups": duplicate_matchups,
        "noncanonical_matchups": noncanonical_matchups,
        "reference_failures": reference_failures,
    }


def _validate_matchup_artifact_references(
    *,
    final_eval_dir: Path,
    matchup: dict[str, Any],
    matchup_index: int,
    reference_failures: list[str],
) -> None:
    for field_name, expected_kind in (
        ("matchup_dir", "directory"),
        ("episodes_path", "file"),
        ("summary_path", "file"),
        ("diagnostics_path", "file"),
        ("posterior_samples_path", "file"),
    ):
        try:
            artifact_path = _require_relative_artifact_path(
                final_eval_dir,
                value=matchup.get(field_name),
                field_name=f"matchups[{matchup_index}].{field_name}",
            )
        except ValueError as exc:
            reference_failures.append(str(exc))
            continue
        exists = artifact_path.is_dir() if expected_kind == "directory" else artifact_path.is_file()
        if not exists:
            reference_failures.append(
                "matchups["
                f"{matchup_index}].{field_name} missing {expected_kind}: "
                f"{artifact_path.relative_to(final_eval_dir).as_posix()}"
            )


__all__ = ["build_final_eval_matchup_contract"]
