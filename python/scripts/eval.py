from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.eval import (
    build_matchup_export,
    build_seat_advantage_diagnostics,
    load_eval_game_records,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.simulator_contract import load_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract, should_verify_runtime_spec_bundle

_SHA256_HEX_LENGTH = 64


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation reporting/contract entrypoint for pre-recorded episodes")
    parser.add_argument(
        "--stack-config",
        type=Path,
        required=True,
        help="Path to the stack config used for contract checks and summary settings",
    )
    parser.add_argument(
        "--spec-hash",
        type=str,
        default="",
        help="Expected compatibility spec hash or full spec bundle SHA-256 for contract validation",
    )
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional startup banner/log label only; not persisted in summary exports",
    )
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=None,
        help="Existing seat-swapped episodes.jsonl to summarize (no rollout generation)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Output path for summary JSON export",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output path for summary CSV export",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        help="Output path for seat diagnostics JSON export",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for uncertainty",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed")
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    if args.episodes_jsonl is None and (args.summary_json is not None or args.summary_csv is not None):
        parser.error("--summary-json/--summary-csv require --episodes-jsonl")

    stack = load_stack_config(args.stack_config)
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    reproducibility = stack.config.reproducibility
    policy = "hard_fail"
    require_export_spec_bundle = False
    persist_in_manifest = False
    if reproducibility is not None:
        spec_bundle_policy = reproducibility.spec_bundle
        policy = "hard_fail" if spec_bundle_policy.fail_on_spec_mismatch else "disabled"
        require_export_spec_bundle = spec_bundle_policy.require_export_spec_bundle
        persist_in_manifest = spec_bundle_policy.persist_in_manifest

    contract = None
    if should_verify_runtime_spec_bundle(
        expected_spec_hash=args.spec_hash,
        require_export_spec_bundle=require_export_spec_bundle,
        persist_in_manifest=persist_in_manifest,
    ):
        contract = load_simulator_contract(stack.root)
        assert_spec_bundle_contract(args.spec_hash, contract.spec_bundle)

    reported_spec_hash = contract.spec_hash256 if contract is not None else "(not checked)"
    print_startup_banner(
        reported_spec_hash,
        config_hash256,
        run_label=run_label,
        spec_mismatch_policy=policy,
    )
    if contract is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={contract.simulator.get('compatibility_hash', '')} sha256={contract.spec_hash256}"
        )

    if args.episodes_jsonl is None:
        print(f"Evaluation contract check complete; no episodes were summarized. Seed sets: {sorted(stack.seed_sets)}")
        return

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    records = load_eval_game_records(args.episodes_jsonl)
    payload = build_matchup_export(
        records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=evaluation.final_matrix_stage2_adaptive_max_paired_seeds,
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary_json = args.summary_json or args.episodes_jsonl.with_suffix(".summary.json")
    summary_csv = args.summary_csv or args.episodes_jsonl.with_suffix(".summary.csv")
    write_matchup_summary_json(summary_json, payload)
    write_matchup_summary_csv(summary_csv, payload)

    print(f"Evaluation summary JSON: {summary_json}")
    print(f"Evaluation summary CSV: {summary_csv}")
    print("Evaluation reports were derived from a pre-recorded episodes file; no rollouts were executed here.")

    if args.diagnostics_json is not None:
        diagnostics_payload = build_seat_advantage_diagnostics(records)
        write_matchup_diagnostics_json(args.diagnostics_json, diagnostics_payload)
        print(f"Evaluation diagnostics JSON: {args.diagnostics_json}")


if __name__ == "__main__":
    main()
