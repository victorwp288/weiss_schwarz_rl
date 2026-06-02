from __future__ import annotations

from pathlib import Path

from weiss_rl.workflows.entrypoint_command_builders import build_eval_entrypoint_command
from weiss_rl.workflows.evaluation_workflow.command_config import EVAL_STACK_CONFIG


def _eval_command(
    *,
    python_exe: str,
    run_dir: Path,
    b1_baseline_run_dir: Path | None,
    smoke: bool,
) -> list[str]:
    policy_ids: tuple[str, ...] = ()
    extra_args: list[str] = []
    if smoke:
        policy_ids = (
            "B0 RandomLegal",
            "B1 NoLeague baseline",
            "B2 HeuristicPublic",
            "B3 HeuristicPublicAggro",
            "B4 HeuristicPublicControl",
        )
        extra_args.extend(
            (
                "--paired-seed-limit",
                "1",
                "--stage1-paired-seeds",
                "1",
                "--max-paired-seeds",
                "1",
                "--bootstrap-samples",
                "16",
                "--skip-metagame",
                "--skip-figures",
                "--skip-readiness",
            )
        )
    return build_eval_entrypoint_command(
        python_exe=python_exe,
        stack_config=EVAL_STACK_CONFIG,
        run_dir=run_dir,
        path_style="posix",
        b1_baseline_run_dir=b1_baseline_run_dir,
        policy_ids=policy_ids,
        extra_args=extra_args,
    )
