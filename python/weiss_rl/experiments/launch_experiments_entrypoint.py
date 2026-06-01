from __future__ import annotations

import torch

from weiss_rl.experiments.experiment_launcher import build_launch_plan, execute_launch_plan, resolve_devices
from weiss_rl.experiments.launch_entrypoint_runtime import run_launch_entrypoint_main
from weiss_rl.experiments.launch_experiments_cli import (
    build_launch_experiments_parser,
    launch_summary_line,
    run_launch_experiments_from_args,
)


def main() -> None:
    run_launch_entrypoint_main(
        entrypoint_file=__file__,
        torch_module=torch,
        build_parser_fn=build_launch_experiments_parser,
        run_from_args_fn=run_launch_experiments_from_args,
        summary_line_fn=launch_summary_line,
        resolve_devices_fn=resolve_devices,
        build_launch_plan_fn=build_launch_plan,
        execute_launch_plan_fn=execute_launch_plan,
    )


if __name__ == "__main__":
    main()
