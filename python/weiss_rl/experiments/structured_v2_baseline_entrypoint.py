from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.structured_acceptance import build_structured_baseline_contract


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the structured_v2 baseline contract from the healthy post-MainMove-fix run and B2 audit"
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        required=True,
        help="Run directory containing periodic dev eval summaries",
    )
    parser.add_argument(
        "--baseline-update",
        type=int,
        default=300,
        help="Dev-eval update to treat as the frozen baseline (default: 300)",
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        required=True,
        help="Aggregated B2 disagreement audit summary JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <baseline-run-dir>/structured_v2/baseline_contract.json",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.baseline_update <= 0:
        parser.error("--baseline-update must be > 0")

    dev_eval_summary = (
        args.baseline_run_dir / "eval" / "dev_eval" / f"update_{int(args.baseline_update)}" / "summary.json"
    )
    if not dev_eval_summary.is_file():
        parser.error(f"baseline dev-eval summary not found: {dev_eval_summary}")
    if not args.audit_summary.is_file():
        parser.error(f"audit summary not found: {args.audit_summary}")

    contract = build_structured_baseline_contract(
        baseline_run_dir=args.baseline_run_dir,
        baseline_update=int(args.baseline_update),
        dev_eval_summary=json.loads(dev_eval_summary.read_text(encoding="utf-8")),
        audit_summary_path=args.audit_summary,
        audit_summary=json.loads(args.audit_summary.read_text(encoding="utf-8")),
    )
    out_path = (
        args.out_json.resolve()
        if args.out_json is not None
        else (args.baseline_run_dir / "structured_v2" / "baseline_contract.json").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote structured_v2 baseline contract to {out_path}")


if __name__ == "__main__":
    main()
