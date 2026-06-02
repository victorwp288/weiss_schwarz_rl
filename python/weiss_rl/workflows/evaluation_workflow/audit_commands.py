from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.command_config import EVAL_STACK_CONFIG


def _b2_audit_command(
    *,
    python_exe: str,
    run_dir: Path,
    episodes_jsonl: Path,
    policy_id: str,
    output_run_dir: Path | None,
    snapshot_registry_json: Path | None,
    summary_json: Path | None,
    top_k: int,
    top_actions: int,
    allow_policy_id_mismatch: bool,
    accepted_snapshot_config_hashes: Sequence[str],
) -> list[str]:
    resolved_output_run_dir = output_run_dir or (run_dir / "eval" / "b2_disagreement")
    command = [
        python_exe,
        "-m",
        "weiss_rl.diagnostics.b2_disagreement_audit",
        "--stack-config",
        EVAL_STACK_CONFIG.as_posix(),
        "--run-dir",
        run_dir.as_posix(),
        "--output-run-dir",
        resolved_output_run_dir.as_posix(),
        "--episodes-jsonl",
        episodes_jsonl.as_posix(),
        "--policy-id",
        policy_id,
        "--top-k",
        str(top_k),
        "--top-actions",
        str(top_actions),
    ]
    if allow_policy_id_mismatch:
        command.append("--allow-policy-id-mismatch")
    for config_hash in accepted_snapshot_config_hashes:
        command.extend(["--accept-snapshot-config-hash", str(config_hash)])
    if snapshot_registry_json is not None:
        command.extend(["--snapshot-registry-json", snapshot_registry_json.as_posix()])
    if summary_json is not None:
        command.extend(["--summary-json", summary_json.as_posix()])
    return command
