from __future__ import annotations

import argparse
from typing import Any

from weiss_rl.workflows.eval_support.eval_parser_validation import _require_positive_int, _resolve_run_label
from weiss_rl.workflows.eval_support.eval_startup_state import EvalValidatedArgs


def validate_eval_args(*, parser: argparse.ArgumentParser, args: Any) -> EvalValidatedArgs:
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    _require_positive_int(parser, "--bootstrap-samples", args.bootstrap_samples)
    _require_positive_int(parser, "--public-demo-paired-seeds", args.public_demo_paired_seeds)
    _require_positive_int(parser, "--public-demo-bootstrap-samples", args.public_demo_bootstrap_samples)
    paired_seed_limit = _require_positive_int(parser, "--paired-seed-limit", args.paired_seed_limit)
    stage1_paired_seeds = _require_positive_int(parser, "--stage1-paired-seeds", args.stage1_paired_seeds)
    max_paired_seeds = _require_positive_int(parser, "--max-paired-seeds", args.max_paired_seeds)

    if args.public_demo:
        if args.run_dir is None:
            parser.error("--public-demo requires --run-dir")
        if args.episodes_jsonl is not None:
            parser.error("--public-demo cannot be combined with --episodes-jsonl")
    elif not args.skip_readiness and (args.skip_metagame or args.skip_figures):
        parser.error("--skip-metagame or --skip-figures requires --skip-readiness")
    elif args.run_dir is not None and args.episodes_jsonl is not None:
        parser.error("--run-dir cannot be combined with --episodes-jsonl outside --public-demo mode")
    elif args.episodes_jsonl is None and (
        args.summary_json is not None or args.summary_csv is not None or args.diagnostics_json is not None
    ):
        parser.error("--summary-json/--summary-csv/--diagnostics-json require --episodes-jsonl")

    return EvalValidatedArgs(
        run_label=run_label,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )
