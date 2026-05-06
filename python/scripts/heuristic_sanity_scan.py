#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import pathlib
import re
import subprocess

JOBS = [
    ("B3 HeuristicPublicAggro", "B0 RandomLegal"),
    ("B4 HeuristicPublicControl", "B0 RandomLegal"),
    ("B3 HeuristicPublicAggro", "B2 HeuristicPublic"),
    ("B4 HeuristicPublicControl", "B2 HeuristicPublic"),
    ("B3 HeuristicPublicAggro", "B4 HeuristicPublicControl"),
    ("B4 HeuristicPublicControl", "B3 HeuristicPublicAggro"),
]


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run(job: tuple[str, str]) -> tuple[str, int]:
    focal, opponent = job
    logdir = pathlib.Path("logs/heuristic_sanity16")
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{safe(focal)}__vs__{safe(opponent)}.log"
    cmd = [
        ".venv-exp034/bin/python",
        "python/scripts/targeted_confirm_eval.py",
        "--stack-config",
        "configs/presets/eval_gpu_exp031_fast_20260506.yaml",
        "--run-dir",
        "runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506",
        "--snapshot-registry-json",
        "runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/training/snapshots/registry.json",
        "--b1-baseline-run-dir",
        "runs/exp-002-current-spec-b1-noleague-baseline",
        "--focal-policy-id",
        focal,
        "--opponent",
        opponent,
        "--paired-seeds",
        "16",
        "--workers",
        "1",
        "--output-subdir",
        "heuristic_sanity16",
    ]
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
