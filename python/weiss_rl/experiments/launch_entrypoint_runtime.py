from __future__ import annotations

from pathlib import Path
from typing import Any


def entrypoint_repo_root(entrypoint_file: str | Path) -> Path:
    return Path(entrypoint_file).resolve().parents[3].parent


def torch_cuda_state(torch_module: Any) -> tuple[bool, int]:
    return bool(torch_module.cuda.is_available()), int(torch_module.cuda.device_count())


def run_launch_entrypoint_main(
    *,
    entrypoint_file: str | Path,
    torch_module: Any,
    build_parser_fn: Any,
    run_from_args_fn: Any,
    summary_line_fn: Any,
    resolve_devices_fn: Any,
    build_launch_plan_fn: Any,
    execute_launch_plan_fn: Any,
    print_fn: Any = print,
) -> None:
    parser = build_parser_fn()
    args = parser.parse_args()
    cuda_available, cuda_count = torch_cuda_state(torch_module)
    summary = run_from_args_fn(
        args,
        repo_root=entrypoint_repo_root(entrypoint_file),
        cuda_available=cuda_available,
        cuda_count=cuda_count,
        resolve_devices_fn=resolve_devices_fn,
        build_launch_plan_fn=build_launch_plan_fn,
        execute_launch_plan_fn=execute_launch_plan_fn,
    )
    print_fn(summary_line_fn(summary))


def run_sweep_entrypoint_main(
    *,
    entrypoint_file: str | Path,
    torch_module: Any,
    build_parser_fn: Any,
    run_from_args_fn: Any,
    summary_line_fn: Any,
    resolve_devices_fn: Any,
    build_sweep_launch_plan_fn: Any,
    execute_launch_plan_fn: Any,
    print_fn: Any = print,
) -> None:
    parser = build_parser_fn()
    args = parser.parse_args()
    cuda_available, cuda_count = torch_cuda_state(torch_module)
    summary, plan_path = run_from_args_fn(
        args,
        repo_root=entrypoint_repo_root(entrypoint_file),
        cuda_available=cuda_available,
        cuda_count=cuda_count,
        resolve_devices_fn=resolve_devices_fn,
        build_sweep_launch_plan_fn=build_sweep_launch_plan_fn,
        execute_launch_plan_fn=execute_launch_plan_fn,
    )
    print_fn(summary_line_fn(preset=str(args.preset), summary=summary, plan_path=plan_path))


__all__ = [
    "entrypoint_repo_root",
    "run_launch_entrypoint_main",
    "run_sweep_entrypoint_main",
    "torch_cuda_state",
]
