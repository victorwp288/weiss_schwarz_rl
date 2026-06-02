from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import hash_seed_file


def god_search_payload_from_args(args: Any) -> dict[str, Any] | None:
    mode = str(getattr(args, "god_search_mode", "disabled") or "disabled").strip()
    if mode == "disabled":
        return None
    return {
        "mode": mode,
        "top_k": int(getattr(args, "god_search_top_k", 4)),
        "rollouts_per_action": int(getattr(args, "god_search_rollouts_per_action", 1)),
        "max_rollout_decisions": int(getattr(args, "god_search_max_rollout_decisions", 0)),
        "max_search_decisions_per_game": int(getattr(args, "god_search_max_search_decisions_per_game", 0)),
        "rollout_policy": str(getattr(args, "god_search_rollout_policy", "eval") or "eval"),
        "apply_to_focal_only": True,
        "verify_prefix_replay": not bool(getattr(args, "god_search_no_prefix_verify", False)),
        "fail_on_prefix_mismatch": not bool(getattr(args, "god_search_soft_prefix_fail", False)),
        "trace_limit": int(getattr(args, "god_search_trace_limit", 24)),
    }


def targeted_worker_summary_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    summary_payload = result["summary"]
    summary = summary_payload["summary"]
    uncertainty = summary_payload["uncertainty"]
    games = int(summary.get("games", 0))
    wins = int(summary.get("wins", 0))
    return {
        "focal_policy_id": result["focal_policy_id"],
        "opponent_policy_id": result["opponent_policy_id"],
        "paired_seeds": int(uncertainty.get("paired_seed_count", len(result.get("used_paired_seeds", ())))),
        "games": games,
        "wins": wins,
        "losses": int(summary.get("losses", 0)),
        "draws": int(summary.get("draws", 0)),
        "mean": float(uncertainty.get("mean", wins / games if games else 0.0)),
        "ci_low": float(uncertainty.get("ci_low", 0.0)),
        "ci_high": float(uncertainty.get("ci_high", 0.0)),
        "prob_gt_half": float(uncertainty.get("prob_gt_half", 0.0)),
        "truncations": int(summary.get("truncations", 0)),
        "engine_errors": int(summary.get("engine_errors", 0)),
        "summary_path": (result["matchup_dir"] / "matchup_summary.json").as_posix(),
        "diagnostics_path": (result["matchup_dir"] / "diagnostics.json").as_posix(),
    }


def build_targeted_confirm_summary(
    *,
    plan: Any,
    results_by_opp: Mapping[str, Mapping[str, Any]],
    started_unix: float,
) -> dict[str, Any]:
    args = plan.args
    rows: list[dict[str, Any]] = []
    for job in plan.jobs:
        opponent = str(job["opponent_policy_id"])
        result = results_by_opp[opponent]
        rows.append(
            {
                "focal_policy_id": args.focal_policy_id,
                "opponent_policy_id": opponent,
                "paired_seeds": result.get("paired_seeds"),
                "games": result.get("games"),
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "draws": result.get("draws"),
                "mean": result.get("mean"),
                "ci_low": result.get("ci_low"),
                "ci_high": result.get("ci_high"),
                "prob_gt_half": result.get("prob_gt_half"),
                "truncations": result.get("truncations"),
                "engine_errors": result.get("engine_errors"),
                "summary_path": result.get("summary_path"),
                "diagnostics_path": result.get("diagnostics_path"),
            }
        )

    anchor_rows = rows[:5]
    league_rows = rows[5:]
    summary: dict[str, Any] = {
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started_unix,
        "focal_policy_id": args.focal_policy_id,
        "output_dir": plan.out_dir.as_posix(),
        "paired_seeds": int(args.paired_seeds),
        "seed_file": {
            "path": plan.seed_file_path.as_posix(),
            "sha256": hash_seed_file(plan.seed_file_path),
            "source": plan.seed_source,
        },
        "games_per_row": int(args.paired_seeds) * 2,
        "stack_config": args.stack_config.as_posix(),
        "eval_sampling_algorithm": plan.eval_sampling_algorithm,
        "model_sampling_temperature": plan.model_sampling_temperature,
        "god_search": god_search_payload_from_args(args),
        "rows": rows,
        "overall": {"wins": sum(row["wins"] for row in rows), "games": sum(row["games"] for row in rows)},
        "anchor_subset": {
            "wins": sum(row["wins"] for row in anchor_rows),
            "games": sum(row["games"] for row in anchor_rows),
        },
        "legacy_subset": {
            "wins": sum(row["wins"] for row in league_rows),
            "games": sum(row["games"] for row in league_rows),
        },
    }
    for key in ("overall", "anchor_subset", "legacy_subset"):
        summary[key]["mean"] = summary[key]["wins"] / summary[key]["games"] if summary[key]["games"] else None
    return summary


def write_targeted_confirm_summary(
    *,
    plan: Any,
    results_by_opp: Mapping[str, Mapping[str, Any]],
    started_unix: float,
) -> tuple[Path, dict[str, Any]]:
    summary = build_targeted_confirm_summary(
        plan=plan,
        results_by_opp=results_by_opp,
        started_unix=started_unix,
    )
    summary_path = plan.out_dir / f"targeted_confirm{plan.args.paired_seeds}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path, summary


__all__ = [
    "build_targeted_confirm_summary",
    "god_search_payload_from_args",
    "targeted_worker_summary_from_result",
    "write_targeted_confirm_summary",
]
