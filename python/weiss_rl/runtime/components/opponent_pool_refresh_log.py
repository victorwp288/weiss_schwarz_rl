"""JSONL refresh records for the runtime opponent pool."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from weiss_rl.runtime.components.opponents import (
    configured_hard_negative_focus_policy_ids,
    configured_row_deficit_policy_weights,
)


def write_opponent_pool_refresh_record(
    *,
    run_dir: Path | str | None,
    league_config: object | None,
    current_update: int,
    registry_path: Path | None,
    candidate_ids: Sequence[str],
    champion_ids: Sequence[str],
    recent_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    resident_policy_ids: Sequence[str],
    loaded_model_ids: Sequence[str],
    stale_demoted: Sequence[str],
    quarantined_count: int,
    reason: str,
) -> None:
    """Append one opponent-pool composition record under the run logs."""
    if run_dir is None:
        return
    resolved_run_dir = Path(run_dir)
    logs_dir = resolved_run_dir / "training" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = build_opponent_pool_refresh_record(
        run_dir=resolved_run_dir,
        league_config=league_config,
        current_update=current_update,
        registry_path=registry_path,
        candidate_ids=candidate_ids,
        champion_ids=champion_ids,
        recent_ids=recent_ids,
        hard_negative_ids=hard_negative_ids,
        resident_policy_ids=resident_policy_ids,
        loaded_model_ids=loaded_model_ids,
        stale_demoted=stale_demoted,
        quarantined_count=quarantined_count,
        reason=reason,
    )
    with (logs_dir / "opponent_pool.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_opponent_pool_refresh_record(
    *,
    run_dir: Path,
    league_config: object | None,
    current_update: int,
    registry_path: Path | None,
    candidate_ids: Sequence[str],
    champion_ids: Sequence[str],
    recent_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    resident_policy_ids: Sequence[str],
    loaded_model_ids: Sequence[str],
    stale_demoted: Sequence[str],
    quarantined_count: int,
    reason: str,
) -> dict[str, object]:
    """Build the machine-readable opponent-pool refresh payload."""
    sampling_config = getattr(league_config, "sampling", league_config)
    return {
        "kind": "opponent_pool_refresh_v1",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": int(os.getpid()),
        "reason": str(reason),
        "update": int(current_update),
        "registry_path": pool_log_path_text(registry_path, run_dir=run_dir),
        "candidate_ids": list(candidate_ids),
        "champion_ids": list(champion_ids),
        "recent_ids": list(recent_ids),
        "hard_negative_ids": list(hard_negative_ids),
        "hard_negative_focus_policy_ids": list(
            configured_hard_negative_focus_policy_ids(league_config=league_config)
        ),
        "hard_negative_focus_weight_multiplier": float(
            getattr(sampling_config, "hard_negative_focus_weight_multiplier", 1.0)
        ),
        "row_deficit_policy_weights": [
            [str(policy_id), float(weight)]
            for policy_id, weight in configured_row_deficit_policy_weights(league_config=league_config)
        ],
        "hard_negative_overlaps_champions": bool(
            getattr(sampling_config, "hard_negative_overlaps_champions", False)
        ),
        "resident_policy_ids": list(resident_policy_ids),
        "loaded_model_ids": list(loaded_model_ids),
        "stale_demoted": list(stale_demoted),
        "quarantined_count": int(quarantined_count),
        "pool_size": int(len(candidate_ids)),
        "champion_pool_size": int(len(champion_ids)),
        "recent_pool_size": int(len(recent_ids)),
        "hard_negative_pool_size": int(len(hard_negative_ids)),
    }


def pool_log_path_text(path: Path | None, *, run_dir: Path) -> str | None:
    """Format a path relative to the run root when possible."""
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


__all__ = [
    "build_opponent_pool_refresh_record",
    "pool_log_path_text",
    "write_opponent_pool_refresh_record",
]
