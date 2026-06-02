from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weiss_rl.workflows.thesis_wrapper_support.execution import _summary_path
from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperResult


def thesis_wrapper_summary_payload(result: ThesisWrapperResult) -> dict[str, Any]:
    plan = result.plan
    return {
        "kind": "thesis_run_wrapper_v1",
        "run_label": plan.run_label,
        "run_dir": plan.run_dir.as_posix(),
        "stack_config": plan.stack_config.as_posix(),
        "eval_stack_config": plan.eval_stack_config.as_posix(),
        "preset": plan.preset,
        "eval_preset": plan.eval_preset,
        "b1_baseline_run_dir": None
        if plan.b1_baseline_run_dir is None
        else plan.b1_baseline_run_dir.resolve().as_posix(),
        "dry_run": plan.dry_run,
        "status": result.status,
        "steps": result.steps,
    }


def write_thesis_wrapper_summary(result: ThesisWrapperResult) -> Path:
    summary_path = _summary_path(result.plan.repo_root, run_label=result.plan.run_label, dry_run=result.plan.dry_run)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(thesis_wrapper_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


__all__ = ["thesis_wrapper_summary_payload", "write_thesis_wrapper_summary"]
