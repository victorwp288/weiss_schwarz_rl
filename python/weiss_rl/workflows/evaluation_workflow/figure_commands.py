from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def _figures_command(
    *,
    python_exe: str,
    run_dir: Path,
    fig_id: str,
    formats: Sequence[str],
) -> list[str]:
    command = [python_exe, "-m", "weiss_rl.workflows.figures.figures_entrypoint", "--run-dir", run_dir.as_posix()]
    if str(fig_id).strip():
        command.extend(["--fig-id", str(fig_id).strip()])
    for fmt in formats:
        command.extend(["--format", str(fmt)])
    return command
