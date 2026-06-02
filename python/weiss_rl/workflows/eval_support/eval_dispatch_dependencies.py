from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.eval import (
    build_matchup_export,
    build_seat_advantage_diagnostics,
    load_eval_game_records,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.experiments.toy_public_demo import (
    public_demo_spec_bundle,
    public_demo_stop_rules,
    run_public_demo_final_eval,
)
from weiss_rl.workflows.eval_canonical import run_canonical_eval_pipeline
from weiss_rl.workflows.eval_support.eval_public_demo_mode import run_public_demo_eval_mode
from weiss_rl.workflows.eval_support.eval_summary_mode import run_summary_only_eval_mode


@dataclass(frozen=True, slots=True)
class EvalDispatchDependencies:
    public_demo_spec_bundle_fn: Any = public_demo_spec_bundle
    public_demo_stop_rules_fn: Any = public_demo_stop_rules
    run_public_demo_final_eval_fn: Any = run_public_demo_final_eval
    run_public_demo_eval_mode_fn: Any = run_public_demo_eval_mode
    run_canonical_eval_pipeline_fn: Any = run_canonical_eval_pipeline
    run_summary_only_eval_mode_fn: Any = run_summary_only_eval_mode
    load_eval_game_records_fn: Any = load_eval_game_records
    build_matchup_export_fn: Any = build_matchup_export
    build_seat_advantage_diagnostics_fn: Any = build_seat_advantage_diagnostics
    write_matchup_diagnostics_json_fn: Any = write_matchup_diagnostics_json
    write_matchup_summary_csv_fn: Any = write_matchup_summary_csv
    write_matchup_summary_json_fn: Any = write_matchup_summary_json


def build_eval_dispatch_dependencies(entrypoint_globals: Mapping[str, Any]) -> EvalDispatchDependencies:
    return EvalDispatchDependencies(
        public_demo_spec_bundle_fn=entrypoint_globals["public_demo_spec_bundle"],
        public_demo_stop_rules_fn=entrypoint_globals["public_demo_stop_rules"],
        run_public_demo_final_eval_fn=entrypoint_globals["run_public_demo_final_eval"],
        run_public_demo_eval_mode_fn=entrypoint_globals["run_public_demo_eval_mode"],
        run_canonical_eval_pipeline_fn=entrypoint_globals["_run_canonical_eval_pipeline"],
        run_summary_only_eval_mode_fn=entrypoint_globals["run_summary_only_eval_mode"],
        load_eval_game_records_fn=entrypoint_globals["load_eval_game_records"],
        build_matchup_export_fn=entrypoint_globals["build_matchup_export"],
        build_seat_advantage_diagnostics_fn=entrypoint_globals["build_seat_advantage_diagnostics"],
        write_matchup_diagnostics_json_fn=entrypoint_globals["write_matchup_diagnostics_json"],
        write_matchup_summary_csv_fn=entrypoint_globals["write_matchup_summary_csv"],
        write_matchup_summary_json_fn=entrypoint_globals["write_matchup_summary_json"],
    )


__all__ = ["EvalDispatchDependencies", "build_eval_dispatch_dependencies"]
