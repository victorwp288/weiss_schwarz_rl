from __future__ import annotations

from weiss_rl.plotting.benchmark_figures import render_benchmark_figures
from weiss_rl.workflows.compare_runs.compare_runs_cli import (
    CompareRunsRequest,
    build_compare_runs_parser,
    compare_runs_request_from_args,
    compare_summary_line,
    run_compare_from_args,
    run_compare_request,
)


def main() -> None:
    parser = build_compare_runs_parser()
    args = parser.parse_args()
    try:
        output_count, out_dir = run_compare_from_args(args, render_benchmark_figures_fn=render_benchmark_figures)
    except ValueError as exc:
        parser.error(str(exc))
    print(compare_summary_line(output_count=output_count, out_dir=out_dir))


__all__ = [
    "CompareRunsRequest",
    "build_compare_runs_parser",
    "compare_runs_request_from_args",
    "compare_summary_line",
    "main",
    "render_benchmark_figures",
    "run_compare_from_args",
    "run_compare_request",
]


if __name__ == "__main__":
    main()
