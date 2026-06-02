from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RenderBenchmarkFiguresFn(Protocol):
    def __call__(self, *, run_dirs: list[Path], out_dir: Path) -> Sequence[Path]: ...


@dataclass(frozen=True, slots=True)
class CompareRunsRequest:
    run_dirs: tuple[Path, ...]
    out_dir: Path


def build_compare_runs_parser() -> argparse.ArgumentParser:
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
    return parser


def compare_runs_request_from_args(args: argparse.Namespace) -> CompareRunsRequest:
    run_dirs = resolve_compare_run_dirs(
        run_dir_values=args.run_dir,
        launch_group_summary=args.launch_group_summary,
    )
    if not run_dirs:
        raise ValueError("At least one --run-dir or --launch-group-summary is required")
    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_dirs[0] / "figures" / "benchmark_compare"
    return CompareRunsRequest(run_dirs=tuple(run_dirs), out_dir=out_dir)


def resolve_compare_run_dirs(
    *,
    run_dir_values: Sequence[str] | None,
    launch_group_summary: Path | None,
) -> list[Path]:
    run_dirs: list[Path] = [Path(path).resolve() for path in run_dir_values or ()]
    if launch_group_summary is not None:
        payload = json.loads(launch_group_summary.read_text(encoding="utf-8"))
        for job in payload.get("jobs", []):
            expected_run_dir = job.get("expected_run_dir")
            if isinstance(expected_run_dir, str) and expected_run_dir.strip():
                run_dirs.append(Path(expected_run_dir).resolve())
    return unique_paths(run_dirs)


def unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def run_compare_from_args(
    args: argparse.Namespace,
    *,
    render_benchmark_figures_fn: RenderBenchmarkFiguresFn,
) -> tuple[int, Path]:
    return run_compare_request(
        compare_runs_request_from_args(args),
        render_benchmark_figures_fn=render_benchmark_figures_fn,
    )


def run_compare_request(
    request: CompareRunsRequest,
    *,
    render_benchmark_figures_fn: RenderBenchmarkFiguresFn,
) -> tuple[int, Path]:
    outputs = render_benchmark_figures_fn(run_dirs=list(request.run_dirs), out_dir=request.out_dir)
    return len(outputs), request.out_dir


def compare_summary_line(*, output_count: int, out_dir: Path) -> str:
    return f"Wrote {output_count} comparison artifacts to {out_dir}"


__all__ = [
    "CompareRunsRequest",
    "RenderBenchmarkFiguresFn",
    "build_compare_runs_parser",
    "compare_runs_request_from_args",
    "compare_summary_line",
    "resolve_compare_run_dirs",
    "run_compare_from_args",
    "run_compare_request",
    "unique_paths",
]
