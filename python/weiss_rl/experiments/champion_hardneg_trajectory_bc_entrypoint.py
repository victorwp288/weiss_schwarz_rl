from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed
from weiss_rl.config import load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.experiments.champion_hardneg_trajectory_bc import (
    build_champion_hardneg_trajectory_bc_dataset,
    normalize_explicit_paired_seeds,
    normalize_include_outcomes,
    write_dataset_summary,
)
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.replay.trajectory_bc import save_replay_trajectory_bc_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build replay trajectory BC data from focal wins against imported champions/hard negatives"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-registry-json", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, default=None)
    parser.add_argument("--focal-policy-id", required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        default=None,
        help="Opponent policy id. Defaults to champion_snapshots from the registry.",
    )
    parser.add_argument(
        "--hard-negative-policy-id",
        action="append",
        default=None,
        help="Mark an opponent as a hard-negative source in the generated metadata.",
    )
    parser.add_argument("--paired-seeds", type=int, default=16)
    parser.add_argument(
        "--paired-seed",
        action="append",
        default=None,
        help="Explicit paired seed to capture. Repeatable; when provided, overrides --paired-seeds/--seed-set.",
    )
    parser.add_argument(
        "--include-outcome",
        action="append",
        default=None,
        help="Focal outcome to retain in the dataset: W/L/D/T. Repeatable; default W. Pass ALL to keep all.",
    )
    parser.add_argument("--seed-set", default="report_eval")
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        require_fixed_python_hash_seed("champion/hard-negative trajectory BC dataset")
    except RuntimeError as err:
        parser.error(str(err))
    if int(args.paired_seeds) <= 0 and not args.paired_seed:
        parser.error("--paired-seeds must be positive")
    if not args.snapshot_registry_json.is_file():
        parser.error(f"snapshot registry not found: {args.snapshot_registry_json}")

    stack = load_stack_config(args.stack_config)
    spec_hash_path = args.run_dir / "spec_hash256.txt"
    if not spec_hash_path.is_file():
        parser.error(f"spec_hash256.txt not found in run dir: {spec_hash_path}")
    contract = load_verified_simulator_contract(
        stack.root,
        expected_spec_hash=spec_hash_path.read_text(encoding="utf-8").strip(),
    )
    registry = SnapshotRegistry.load(args.snapshot_registry_json)
    opponents = tuple(str(item).strip() for item in (args.opponent or registry.champion_snapshots) if str(item).strip())
    if not opponents:
        parser.error("no opponents provided and registry has no champion_snapshots")
    include_outcomes = normalize_include_outcomes(args.include_outcome)
    try:
        explicit_paired_seeds = normalize_explicit_paired_seeds(args.paired_seed)
    except ValueError as err:
        parser.error(str(err))

    dataset, summary = build_champion_hardneg_trajectory_bc_dataset(
        stack=stack,
        contract=contract,
        stack_config=args.stack_config,
        run_dir=args.run_dir,
        output_run_dir=args.output_run_dir,
        output_dataset=args.output,
        snapshot_registry_json=args.snapshot_registry_json,
        focal_policy_id=str(args.focal_policy_id),
        opponent_policy_ids=opponents,
        paired_seed_count=int(args.paired_seeds),
        include_outcomes=include_outcomes,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        hard_negative_policy_ids=tuple(args.hard_negative_policy_id or ()),
        seed_set_name=str(args.seed_set),
        explicit_paired_seeds=explicit_paired_seeds or None,
    )
    save_replay_trajectory_bc_dataset(args.output, dataset)
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    write_dataset_summary(summary_path, summary, dataset)
    print(
        "Champion/hard-negative trajectory BC dataset written to "
        f"{args.output} with {dataset.metadata['train_rows']} train rows across "
        f"{dataset.metadata['bundle_count']} bundles; summary written to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
