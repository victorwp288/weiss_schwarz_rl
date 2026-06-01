from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed
from weiss_rl.config import load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.experiments.champion_hardneg_trajectory_bc import normalize_include_outcomes
from weiss_rl.experiments.paired_flip_trajectory_bc import (
    PairedFlipTrajectoryBcConfig,
    build_paired_flip_trajectory_bc_dataset,
    write_paired_flip_trajectory_bc_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build replay trajectory BC data by rerunning exact paired-flip target seeds"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-registry-json", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--focal-policy-id", required=True)
    parser.add_argument("--paired-flip-targets-json", type=Path, required=True)
    parser.add_argument(
        "--hard-negative-policy-id",
        action="append",
        default=None,
        help="Mark an opponent as a hard-negative source in generated metadata.",
    )
    parser.add_argument(
        "--include-outcome",
        action="append",
        default=None,
        help="Focal outcome to retain in the dataset: W/L/D/T. Repeatable; default W. Pass ALL to keep all.",
    )
    parser.add_argument("--source-label-prefix", default="")
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        require_fixed_python_hash_seed("paired-flip trajectory BC dataset")
    except RuntimeError as err:
        parser.error(str(err))
    if not args.snapshot_registry_json.is_file():
        parser.error(f"snapshot registry not found: {args.snapshot_registry_json}")
    if not args.paired_flip_targets_json.is_file():
        parser.error(f"paired flip targets JSON not found: {args.paired_flip_targets_json}")

    stack = load_stack_config(args.stack_config)
    spec_hash_path = args.run_dir / "spec_hash256.txt"
    if not spec_hash_path.is_file():
        parser.error(f"spec_hash256.txt not found in run dir: {spec_hash_path}")
    contract = load_verified_simulator_contract(
        stack.root,
        expected_spec_hash=spec_hash_path.read_text(encoding="utf-8").strip(),
    )
    include_outcomes = normalize_include_outcomes(args.include_outcome)
    dataset, summary = build_paired_flip_trajectory_bc_dataset(
        PairedFlipTrajectoryBcConfig(
            stack=stack,
            contract=contract,
            stack_config=args.stack_config,
            run_dir=args.run_dir,
            snapshot_registry_json=args.snapshot_registry_json,
            paired_flip_targets_json=args.paired_flip_targets_json,
            focal_policy_id=str(args.focal_policy_id),
            output_run_dir=args.output_run_dir,
            output_dataset=args.output,
            include_outcomes=include_outcomes,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            hard_negative_policy_ids=tuple(args.hard_negative_policy_id or ()),
            source_label_prefix=str(args.source_label_prefix),
        )
    )
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    write_paired_flip_trajectory_bc_summary(summary_path, summary=summary, dataset=dataset)
    print(
        "Paired-flip trajectory BC dataset written to "
        f"{args.output} with {dataset.metadata['train_rows']} train rows across "
        f"{dataset.metadata['bundle_count']} bundles; summary written to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
