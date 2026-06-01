from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from weiss_rl.eval import (
    build_matchup_export,
    build_seat_advantage_diagnostics,
    load_eval_game_records,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme


def run_summary_only_eval_mode(
    *,
    stack: Any,
    episodes_jsonl: Path,
    summary_json: Path | None,
    summary_csv: Path | None,
    diagnostics_json: Path | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    load_eval_game_records_fn: Any = load_eval_game_records,
    build_matchup_export_fn: Any = build_matchup_export,
    build_seat_advantage_diagnostics_fn: Any = build_seat_advantage_diagnostics,
    write_matchup_diagnostics_json_fn: Any = write_matchup_diagnostics_json,
    write_matchup_summary_csv_fn: Any = write_matchup_summary_csv,
    write_matchup_summary_json_fn: Any = write_matchup_summary_json,
) -> None:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    records = load_eval_game_records_fn(episodes_jsonl)
    payload = build_matchup_export_fn(
        records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=evaluation.final_matrix_stage2_adaptive_max_paired_seeds,
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=bootstrap_samples,
        seed=bootstrap_seed,
    )
    resolved_summary_json = summary_json or episodes_jsonl.with_suffix(".summary.json")
    resolved_summary_csv = summary_csv or episodes_jsonl.with_suffix(".summary.csv")
    write_matchup_summary_json_fn(resolved_summary_json, payload)
    write_matchup_summary_csv_fn(resolved_summary_csv, payload)

    print(f"Evaluation summary JSON: {resolved_summary_json}")
    print(f"Evaluation summary CSV: {resolved_summary_csv}")
    print("Evaluation reports were derived from a pre-recorded episodes file; no rollouts were executed here.")

    if diagnostics_json is not None:
        diagnostics_payload = build_seat_advantage_diagnostics_fn(records)
        write_matchup_diagnostics_json_fn(diagnostics_json, diagnostics_payload)
        print(f"Evaluation diagnostics JSON: {diagnostics_json}")


__all__ = ["run_summary_only_eval_mode"]
