from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.harness import EvalGameRecord, ReplaySampleResult, write_episodes_jsonl


def write_final_eval_artifacts(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    layout = maybe_layout(output_dir)
    metadata_path = layout.final_eval_metadata_json() if layout is not None else output_dir / "metadata.json"
    policy_set_path = layout.final_eval_policy_set_json() if layout is not None else output_dir / "policy_set.json"
    summary_path = layout.final_eval_summary_json() if layout is not None else output_dir / "summary.json"
    posterior_samples_json_path = (
        layout.final_eval_posterior_samples_json() if layout is not None else output_dir / "posterior_samples.json"
    )

    metadata_path.write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_set_path.write_text(
        json.dumps({"policy_ids": payload["policy_ids"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    posterior_samples_json_path.write_text(
        json.dumps(payload["posterior_samples"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if layout is not None:
        posterior_payload = cast(Mapping[str, Any], payload["posterior_samples"])
        np.savez_compressed(
            layout.final_eval_posterior_samples_npz(),
            values=np.asarray(posterior_payload.get("values", ()), dtype=np.float64),
            policy_ids=np.asarray(posterior_payload.get("policy_ids", ()), dtype=object),
        )

    matrices_dir = layout.final_eval_matrices_dir if layout is not None else output_dir / "matrices"
    matrices = cast(Mapping[str, Mapping[str, Any]], payload["matrices"])
    for field, matrix_payload in matrices.items():
        json_path = matrices_dir / f"{field}.json"
        csv_path = matrices_dir / f"{field}.csv"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_matrix_csv(csv_path, matrix_payload)
        if layout is not None:
            legacy_name = legacy_payoff_matrix_name(field)
            if legacy_name is not None:
                write_matrix_csv(layout.final_eval_payoff_matrix_csv(legacy_name), matrix_payload)

    manifest_rows = final_eval_matchup_manifest_rows(output_dir=output_dir, matchup_results=matchup_results)
    manifest_path = layout.final_eval_matchups_csv() if layout is not None else output_dir / "matchups.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    aggregate_records = [
        record for result in matchup_results for record in cast(Sequence[EvalGameRecord], result.get("records", ()))
    ]
    if aggregate_records:
        episodes_path = layout.final_eval_episodes_jsonl() if layout is not None else output_dir / "episodes.jsonl"
        write_episodes_jsonl(episodes_path, aggregate_records)
    if layout is not None:
        write_run_level_diagnostics(layout=layout, policy_ids=payload["policy_ids"], matchup_results=matchup_results)
        write_artifact_hashes(layout=layout)


def final_eval_matchup_manifest_rows(
    *,
    output_dir: Path,
    matchup_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "focal_policy_id": result["focal_policy_id"],
            "opponent_policy_id": result["opponent_policy_id"],
            "matchup_dir": relative_to(Path(result["matchup_dir"]), root=output_dir),
            "paired_seed_count": result["summary"]["paired_seeds"],
            "observed_paired_seed_count": result["summary"]["observed_paired_seeds"],
            "excluded_paired_seed_count": result["summary"]["excluded_paired_seeds"],
            "has_payoff_samples": result["summary"]["has_payoff_samples"],
            "stop_reason": result["summary"]["stop_reason"],
        }
        for result in matchup_results
    ]


def write_matrix_csv(path: Path, matrix_payload: Mapping[str, Any]) -> None:
    policy_ids = cast(list[str], matrix_payload["policy_ids"])
    values = cast(list[list[Any]], matrix_payload["values"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, values, strict=True):
            writer.writerow([focal_policy_id, *row])


def legacy_payoff_matrix_name(field: str) -> str | None:
    if field == "mean":
        return "p_mean"
    return None


def maybe_layout(output_dir: Path) -> ArtifactLayout | None:
    try:
        return ArtifactLayout.from_final_eval_dir(output_dir)
    except ValueError:
        return None


def write_run_level_diagnostics(
    *,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
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


def write_artifact_hashes(*, layout: ArtifactLayout) -> None:
    tracked_paths = [
        layout.final_eval_summary_json(),
        layout.final_eval_policy_set_json(),
        layout.final_eval_metadata_json(),
        layout.final_eval_matchups_csv(),
        layout.final_eval_posterior_samples_json(),
        layout.final_eval_posterior_samples_npz(),
        layout.final_eval_matrix_csv("mean"),
        layout.seat_bias_json(),
        layout.truncation_heatmap_csv(),
        layout.replay_verification_json(),
        layout.replay_index_json(),
    ]
    payload = {
        "kind": "final_eval_artifact_hashes_v1",
        "artifacts": {layout.relative(path): sha256_file(path) for path in tracked_paths if path.is_file()},
    }
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "build_run_level_seat_bias_payload",
    "final_eval_matchup_manifest_rows",
    "legacy_payoff_matrix_name",
    "maybe_layout",
    "relative_to",
    "sha256_file",
    "write_artifact_hashes",
    "write_final_eval_artifacts",
    "write_matrix_csv",
    "write_replay_diagnostics",
    "write_run_level_diagnostics",
    "write_truncation_heatmap_csv",
]
