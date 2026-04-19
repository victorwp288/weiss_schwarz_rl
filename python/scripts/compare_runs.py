from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.plotting.benchmark_figures import render_benchmark_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cross-run baseline and scaling comparison artifacts")
    parser.add_argument("--run-dir", action="append", default=None, help="Run directory to include in the comparison")
    parser.add_argument(
        "--launch-group-summary",
        type=Path,
        default=None,
        help="Optional runs/launch_groups/<group>/summary.json to expand into run directories automatically",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <first-run>/figures/benchmark_compare",
    )
    args = parser.parse_args()

    run_dirs: list[Path] = [Path(path).resolve() for path in args.run_dir or ()]
    if args.launch_group_summary is not None:
        payload = json.loads(args.launch_group_summary.read_text(encoding="utf-8"))
        for job in payload.get("jobs", []):
            expected_run_dir = job.get("expected_run_dir")
            if isinstance(expected_run_dir, str) and expected_run_dir.strip():
                run_dirs.append(Path(expected_run_dir).resolve())
    if not run_dirs:
        parser.error("At least one --run-dir or --launch-group-summary is required")
    unique_run_dirs: list[Path] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        unique_run_dirs.append(run_dir)
    out_dir = (
        args.out_dir.resolve() if args.out_dir is not None else unique_run_dirs[0] / "figures" / "benchmark_compare"
    )
    outputs = render_benchmark_figures(run_dirs=unique_run_dirs, out_dir=out_dir)
    print(f"Wrote {len(outputs)} comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
