from __future__ import annotations

import argparse
from pathlib import Path

from weiss_rl.config import load_study_config
from weiss_rl.metagame import build_sensitivity_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metagame sensitivity reports from final_eval artifacts")
    parser.add_argument(
        "--study-config",
        type=Path,
        required=True,
        help="Path to the study-only config providing metagame and sensitivity settings",
    )
    parser.add_argument(
        "--final-eval-dir",
        type=Path,
        required=True,
        help="Path to a final_eval artifact directory containing summary.json and matchup episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for sensitivity artifacts (default: <final-eval-dir>/sensitivity)",
    )
    args = parser.parse_args()

    study = load_study_config(args.study_config)

    out_dir = args.out_dir or (args.final_eval_dir / "sensitivity")
    payload = build_sensitivity_report(
        final_eval_dir=args.final_eval_dir,
        out_dir=out_dir,
        metagame_config=study.metagame,
        sensitivity_config=study.sensitivity,
    )
    print(f"Sensitivity summary JSON: {out_dir / 'summary.json'}")
    print(f"Sensitivity cases: {sorted(payload['cases'])}")
    print(f"Sensitivity delta cases: {sorted(payload['deltas'])}")


if __name__ == "__main__":
    main()
