#!/usr/bin/env python3
"""Compatibility facade for the package-owned targeted confirmation eval CLI."""

from __future__ import annotations

# ruff: noqa: F401,I001

from weiss_rl.eval.targeted_confirm.core import (
    _run_targeted_jobs,
    _targeted_worker,
    main,
)
from weiss_rl.eval.targeted_confirm.plan import (
    DEFAULT_OPPONENTS,
    FAST_LOOP_EXACT_PAIRED_SEEDS,
    MAIN_LEAGUE_FULL13_OPPONENTS,
    MAIN_LEAGUE_SENTINEL_OPPONENTS,
    OPPONENT_SETS,
    TargetedConfirmPlan,
    _god_search_payload_from_args,
    _require_exact_opponent_panel,
    _require_fast_loop_gate,
    _resolve_opponents,
    _resolve_paired_seed_file,
    _targeted_eval_job,
    _validate_fast_loop_eval_request,
    build_arg_parser,
    build_targeted_confirm_jobs,
    build_targeted_confirm_summary,
    parse_args,
    prepare_targeted_confirm_plan,
    validate_targeted_confirm_request,
    write_targeted_confirm_summary,
)


if __name__ == "__main__":
    main()
