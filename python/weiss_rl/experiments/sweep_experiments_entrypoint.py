from __future__ import annotations

import torch

from weiss_rl.experiments.experiment_launcher import execute_launch_plan, resolve_devices
from weiss_rl.experiments.launch_entrypoint_runtime import run_sweep_entrypoint_main
from weiss_rl.experiments.sweep_experiments_cli import (
    build_sweep_experiments_parser,
    run_sweep_experiments_from_args,
    sweep_summary_line,
)
from weiss_rl.experiments.sweeps import build_sweep_launch_plan


def main() -> None:
    run_sweep_entrypoint_main(
        entrypoint_file=__file__,
        torch_module=torch,
        build_parser_fn=build_sweep_experiments_parser,
        run_from_args_fn=run_sweep_experiments_from_args,
        summary_line_fn=sweep_summary_line,
        resolve_devices_fn=resolve_devices,
        build_sweep_launch_plan_fn=build_sweep_launch_plan,
        execute_launch_plan_fn=execute_launch_plan,
    )


if __name__ == "__main__":
    main()
