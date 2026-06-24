"""Run-level diagnostics written after canonical final evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.simulator.harness import EvalGameRecord, ReplaySampleResult


def write_run_level_diagnostics(
    *,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    """Write final-eval diagnostics that summarize all matchup artifacts."""
    aggregate_records = [
        record for result in matchup_results for record in cast(Sequence[EvalGameRecord], result.get("records", ()))
    ]
    if aggregate_records:
        layout.seat_bias_json().write_text(
            json.dumps(
                build_run_level_seat_bias_payload(matchup_results=matchup_results),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    write_truncation_heatmap_csv(layout=layout, policy_ids=policy_ids, matchup_results=matchup_results)
    replay_verification_payload = write_replay_diagnostics(layout=layout, matchup_results=matchup_results)
    layout.replay_verification_json().write_text(
        json.dumps(replay_verification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_truncation_heatmap_csv(
    *,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    """Write the policy-by-policy truncation-rate heatmap."""
    result_by_pair = {
        (str(result["focal_policy_id"]), str(result["opponent_policy_id"])): result for result in matchup_results
    }
    layout.truncation_heatmap_csv().parent.mkdir(parents=True, exist_ok=True)
    with layout.truncation_heatmap_csv().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id in policy_ids:
            row: list[Any] = [focal_policy_id]
            for opponent_policy_id in policy_ids:
                key = (focal_policy_id, opponent_policy_id)
                mirror_key = (opponent_policy_id, focal_policy_id)
                result = result_by_pair.get(key) or result_by_pair.get(mirror_key)
                truncations = 0
                games = 0
                if result is not None:
                    summary = cast(Mapping[str, Any], result["summary"]).get("summary", {})
                    truncations = int(cast(Mapping[str, Any], summary).get("truncations", 0))
                    games = int(cast(Mapping[str, Any], summary).get("games", 0))
                row.append((truncations / games) if games else 0.0)
            writer.writerow(row)


def build_run_level_seat_bias_payload(*, matchup_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the retained seat-bias summary for all final-eval matchups."""
    matchup_rows: list[dict[str, Any]] = []
    decisive_games_total = 0
    seat0_wins_total = 0
    for result in matchup_results:
        diagnostics = cast(Mapping[str, Any], result["diagnostics"])
        seat_results = cast(Mapping[str, Any], diagnostics.get("seat_results", {}))
        seat0_wins = int(seat_results.get("seat0_wins", 0))
        seat1_wins = int(seat_results.get("seat1_wins", 0))
        decisive_games = int(seat_results.get("decisive_games", seat0_wins + seat1_wins))
        if decisive_games <= 0:
            continue
        seat0_rate = seat0_wins / decisive_games
        seat1_rate = seat1_wins / decisive_games
        matchup_rows.append(
            {
                "policy_a": str(result["focal_policy_id"]),
                "policy_b": str(result["opponent_policy_id"]),
                "seat0_win_rate": seat0_rate,
                "seat1_win_rate": seat1_rate,
                "decisive_games": decisive_games,
            }
        )
        decisive_games_total += decisive_games
        seat0_wins_total += seat0_wins

    global_seat0_rate = (seat0_wins_total / decisive_games_total) if decisive_games_total else 0.5
    return {
        "kind": "seat_bias_summary_v1",
        "global": {
            "seat0_win_rate": global_seat0_rate,
            "ci_low": global_seat0_rate,
            "ci_high": global_seat0_rate,
            "decisive_games": decisive_games_total,
        },
        "matchups": matchup_rows,
    }


def write_replay_diagnostics(
    *,
    layout: ArtifactLayout,
    matchup_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Write replay index details and return the replay verification summary."""
    replay_samples = [
        sample
        for result in matchup_results
        for sample in cast(Sequence[ReplaySampleResult], result.get("replay_samples", ()))
    ]
    replay_index_payload = {
        "kind": "replay_index_v1",
        "samples": [
            {
                "pair_index": int(sample.pair_index),
                "swap_index": int(sample.swap_index),
                "episode_index": int(sample.episode_index),
                "focal_policy_id": str(sample.focal_policy_id),
                "opponent_policy_id": str(sample.opponent_policy_id),
                "raw_replay_path": sample.raw_replay_path,
                "bundle_path": str(sample.bundle_path),
                "verification_report_path": str(sample.verification_report_path),
                "verification_status": str(sample.verification_status),
                "replay_key64": str(sample.replay_key64),
                "matched": bool(sample.matched),
                "error": sample.error,
            }
            for sample in replay_samples
        ],
    }
    layout.replay_index_json().write_text(
        json.dumps(replay_index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not replay_samples:
        return {
            "kind": "replay_verification_summary_v1",
            "status": "not_sampled",
            "sampled_episode_count": 0,
            "verified_episode_count": 0,
            "failed_episode_count": 0,
            "verified_report_count": 0,
            "message": "final_eval completed without sampled replay captures",
            "index_path": layout.relative(layout.replay_index_json()),
        }

    verified_episode_count = sum(
        1 for sample in replay_samples if sample.matched and str(sample.verification_status) == "success"
    )
    failed_samples = [
        sample for sample in replay_samples if not sample.matched or str(sample.verification_status) != "success"
    ]
    return {
        "kind": "replay_verification_summary_v1",
        "status": "ok" if not failed_samples else "failed",
        "sampled_episode_count": len(replay_samples),
        "verified_episode_count": verified_episode_count,
        "failed_episode_count": len(failed_samples),
        "verified_report_count": len(replay_samples),
        "index_path": layout.relative(layout.replay_index_json()),
        "failed_replays": [
            {
                "replay_key64": str(sample.replay_key64),
                "verification_status": str(sample.verification_status),
                "verification_report_path": str(sample.verification_report_path),
                "error": sample.error,
            }
            for sample in failed_samples
        ],
    }


__all__ = [
    "build_run_level_seat_bias_payload",
    "write_replay_diagnostics",
    "write_run_level_diagnostics",
    "write_truncation_heatmap_csv",
]
