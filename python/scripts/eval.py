from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.eval import (
    build_matchup_export,
    build_seat_advantage_diagnostics,
    load_eval_game_records,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.simulator_contract import load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    parser.add_argument("--episodes-jsonl", type=Path, default=None, help="Existing seat-swapped episodes.jsonl to summarize")
    parser.add_argument("--summary-json", type=Path, default=None, help="Output path for summary JSON export")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Output path for summary CSV export")
    parser.add_argument("--diagnostics-json", type=Path, default=None, help="Output path for seat diagnostics JSON export")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap sample count for uncertainty")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed")
    args = parser.parse_args()

    if args.episodes_jsonl is None and (args.summary_json is not None or args.summary_csv is not None):
        parser.error("--summary-json/--summary-csv require --episodes-jsonl")

    stack = load_stack_config(args.stack_config)
    reproducibility = stack.config.reproducibility
    policy = "hard_fail"
    contract = None
    if reproducibility is not None:
        spec_bundle_policy = reproducibility.spec_bundle
        policy = "hard_fail" if spec_bundle_policy.fail_on_spec_mismatch else "disabled"
        should_verify = (
            bool(args.spec_hash.strip())
            or spec_bundle_policy.require_export_spec_bundle
            or spec_bundle_policy.persist_in_manifest
        )
        if should_verify:
            contract = load_simulator_contract(stack.root)
            assert_spec_bundle_contract(args.spec_hash, contract.spec_bundle)

    print_startup_banner(args.spec_hash, args.config_hash, args.run_id, spec_mismatch_policy=policy)
    if contract is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={contract.simulator.get('compatibility_hash', '')} sha256={contract.spec_hash256}"
        )

    if args.episodes_jsonl is None:
        print(f"Evaluation scaffold ready; seed sets: {sorted(stack.seed_sets)}")
        return

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    records = load_eval_game_records(args.episodes_jsonl)
    payload = build_matchup_export(
        records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=evaluation.final_matrix_stage2_adaptive_max_paired_seeds,
        scheme=evaluation.final_policy_set_selection.folding,
        sample_count=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary_json = args.summary_json or args.episodes_jsonl.with_suffix(".summary.json")
    summary_csv = args.summary_csv or args.episodes_jsonl.with_suffix(".summary.csv")
    write_matchup_summary_json(summary_json, payload)
    write_matchup_summary_csv(summary_csv, payload)

    print(f"Evaluation summary JSON: {summary_json}")
    print(f"Evaluation summary CSV: {summary_csv}")

    if args.diagnostics_json is not None:
        diagnostics_payload = build_seat_advantage_diagnostics(records)
        write_matchup_diagnostics_json(args.diagnostics_json, diagnostics_payload)
        print(f"Evaluation diagnostics JSON: {args.diagnostics_json}")


if __name__ == "__main__":
    main()
