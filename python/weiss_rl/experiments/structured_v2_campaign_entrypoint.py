from __future__ import annotations

# ruff: noqa: F401,I001

import subprocess
import sys
from pathlib import Path

from weiss_rl.experiments.bootstrap_commands import (
    build_b2_disagreement_audit_entrypoint_command,
    build_training_entrypoint_command,
)
from weiss_rl.experiments.structured_v2_campaign_core import (
    DEFAULT_B2_ANCHOR as _DEFAULT_B2_ANCHOR,
    DEFAULT_SEEDS as _DEFAULT_SEEDS,
    build_structured_v2_campaign_parser,
    command_env as _command_env,
    find_counter as _find_counter,
    focal_policy_id_from_dev_eval as _focal_policy_id_from_dev_eval,
    freeze_baseline_contract as _freeze_baseline_contract,
    repo_run_dir as _repo_run_dir,
    run_label as _run_label,
    run_step as _run_step,
    run_structured_v2_audit_step,
    run_structured_v2_campaign_from_args,
    run_structured_v2_campaign_seed,
    summary_path as _summary_path,
    u120_acceptance_payload as _u120_acceptance_payload,
    workspace_root as _workspace_root,
    write_summary as _write_summary,
)

_build_parser = build_structured_v2_campaign_parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2] if args.repo_root is None else Path(args.repo_root).resolve()
    return run_structured_v2_campaign_from_args(
        args,
        repo_root=repo_root,
        python_exe=sys.executable,
        build_training_command_fn=build_training_entrypoint_command,
        build_audit_command_fn=build_b2_disagreement_audit_entrypoint_command,
        subprocess_run_fn=subprocess.run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
