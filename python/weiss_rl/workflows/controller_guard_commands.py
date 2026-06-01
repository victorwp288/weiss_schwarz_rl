from __future__ import annotations

from pathlib import Path


def _guard_run_command(
    *,
    python_exe: str,
    run_dir: Path,
    required_anchors: tuple[str, ...],
    min_latest_anchor_score: float,
    max_latest_drop: float,
    require_promotion_pass_after_attempts: int,
    max_consecutive_promotion_failures: int,
    max_vtrace_rho_p99: float | None,
) -> list[str]:
    command = [
        python_exe,
        "-m",
        "weiss_rl.diagnostics.learning_progress",
        "--run-dir",
        run_dir.as_posix(),
        "--league-guard",
        "--guard-min-latest-anchor-score",
        str(float(min_latest_anchor_score)),
        "--guard-max-latest-drop",
        str(float(max_latest_drop)),
        "--guard-require-promotion-pass-after-attempts",
        str(int(require_promotion_pass_after_attempts)),
        "--guard-max-consecutive-promotion-failures",
        str(int(max_consecutive_promotion_failures)),
    ]
    for anchor in required_anchors:
        command.extend(["--guard-required-anchor", anchor])
    if max_vtrace_rho_p99 is not None:
        command.extend(["--guard-max-vtrace-rho-p99", str(float(max_vtrace_rho_p99))])
    return command
