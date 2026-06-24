#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import pathlib
import re
import subprocess

from weiss_rl.experiments.bootstrap_commands import build_targeted_confirm_entrypoint_command

JOBS = [
    ("B3 HeuristicPublicAggro", "B0 RandomLegal"),
    ("B4 HeuristicPublicControl", "B0 RandomLegal"),
    ("B3 HeuristicPublicAggro", "B2 HeuristicPublic"),
    ("B4 HeuristicPublicControl", "B2 HeuristicPublic"),
    ("B3 HeuristicPublicAggro", "B4 HeuristicPublicControl"),
    ("B4 HeuristicPublicControl", "B3 HeuristicPublicAggro"),
]
_PYTHON_COMMAND = None
_STACK_CONFIG = pathlib.Path("configs/presets/structured_acceptance_standard_thesis_eval.yaml")
_RUN_DIR = pathlib.Path("runs/main_champion_hardneg_interp_u10_repair_a015_20260517")
_B1_BASELINE_RUN_DIR = _RUN_DIR
_OUTPUT_SUBDIR = "heuristic_sanity16"
_PAIRED_SEEDS = 16


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_heuristic_sanity_command(*, focal: str, opponent: str) -> list[str]:
    return build_targeted_confirm_entrypoint_command(
        repo_root=None,
        stack_config=_STACK_CONFIG,
        run_dir=_RUN_DIR,
        b1_baseline_run_dir=_B1_BASELINE_RUN_DIR,
        focal_policy_id=focal,
        paired_seeds=_PAIRED_SEEDS,
        bootstrap_samples=None,
        output_subdir=_OUTPUT_SUBDIR,
        opponents=(opponent,),
        python_command=_PYTHON_COMMAND,
    )


def run(job: tuple[str, str]) -> tuple[str, int]:
    focal, opponent = job
    logdir = pathlib.Path("logs/heuristic_sanity16")
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{safe(focal)}__vs__{safe(opponent)}.log"
    cmd = build_heuristic_sanity_command(focal=focal, opponent=opponent)
    with log.open("w") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    return log.as_posix(), result.returncode


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run, job) for job in JOBS]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
