from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs, thesis_wrapper_inputs_from_args
from weiss_rl.workflows.thesis_wrapper_support.presets import (
    _default_eval_preset_for_preset,
    _resolve_eval_stack_config,
    _resolve_stack_config,
)
from weiss_rl.workflows.thesis_wrapper_support.state import ThesisWrapperRequest


def thesis_wrapper_request(*, args: Any, repo_root: Path, python_exe: str) -> ThesisWrapperRequest:
    inputs = args if isinstance(args, ThesisWrapperInputs) else thesis_wrapper_inputs_from_args(args)
    run_dir = repo_root / "runs" / inputs.run_label
    stack_config = _resolve_stack_config(repo_root=repo_root, stack_config=inputs.stack_config, preset=inputs.preset)
    eval_preset = inputs.eval_preset.strip()
    if not eval_preset and inputs.stack_config is None:
        eval_preset = _default_eval_preset_for_preset(inputs.preset)
    eval_stack_config = _resolve_eval_stack_config(
        repo_root=repo_root,
        eval_stack_config=inputs.eval_stack_config,
        train_stack_config=inputs.stack_config,
        eval_preset=eval_preset,
    )
    return ThesisWrapperRequest(
        inputs=inputs,
        repo_root=repo_root,
        python_exe=python_exe,
        run_dir=run_dir,
        stack_config=stack_config,
        eval_stack_config=eval_stack_config,
        eval_preset=eval_preset,
    )


__all__ = ["thesis_wrapper_request"]
