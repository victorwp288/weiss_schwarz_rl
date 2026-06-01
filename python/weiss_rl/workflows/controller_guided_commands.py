from __future__ import annotations

from pathlib import Path


def _guided_bootstrap_loop_command(
    *,
    python_exe: str,
    initial_run_dir: Path,
    initial_policy_id: str,
    seed_run_dir: Path | None,
    run_prefix: str,
    stack_config: Path,
    alias_policy_id: str,
    segments: int,
    segment_updates: int,
    confirm_paired_seeds: int,
    stop_on_latest_falloff: bool,
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "weiss_rl.experiments.segmented_b1_guided_bootstrap_entrypoint",
        "--initial-run-dir",
        initial_run_dir.as_posix(),
        "--initial-policy-id",
        str(initial_policy_id),
        "--run-prefix",
        str(run_prefix),
        "--stack-config",
        stack_config.as_posix(),
        "--alias-policy-id",
        str(alias_policy_id),
        "--segments",
        str(int(segments)),
        "--segment-updates",
        str(int(segment_updates)),
        "--confirm-paired-seeds",
        str(int(confirm_paired_seeds)),
    ]
    if seed_run_dir is not None:
        command.extend(["--seed-run-dir", seed_run_dir.as_posix()])
    if stop_on_latest_falloff:
        command.append("--stop-on-latest-falloff")
    return command
