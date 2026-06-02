from __future__ import annotations

from weiss_rl.workflows.thesis_wrapper_support.cli import (
    build_thesis_wrapper_parser,
    main,
    print_thesis_wrapper_presets,
    run_thesis_wrapper_cli,
    thesis_wrapper_repo_root,
)
from weiss_rl.workflows.thesis_wrapper_support.commands import (
    _DEFAULT_EVAL_PRESET,
    _DEFAULT_EVAL_PRESET_OVERRIDES,
    _PRESET_PATHS,
    _command_display,
    _default_eval_preset_for_preset,
    _resolve_cli_path,
    _resolve_eval_stack_config,
    _resolve_stack_config,
    _run_step,
    _summary_path,
    build_thesis_compare_command,
    build_thesis_eval_command,
    build_thesis_train_command,
)
from weiss_rl.workflows.thesis_wrapper_support.inputs import ThesisWrapperInputs, thesis_wrapper_inputs_from_args
from weiss_rl.workflows.thesis_wrapper_support.plan import (
    ThesisWrapperCommands,
    ThesisWrapperPlan,
    ThesisWrapperRequest,
    ThesisWrapperResult,
    build_thesis_wrapper_commands,
    build_thesis_wrapper_commands_for_request,
    build_thesis_wrapper_plan,
    build_thesis_wrapper_plan_for_request,
    run_thesis_wrapper_commands,
    run_thesis_wrapper_plan,
    thesis_wrapper_commands_for_plan,
    thesis_wrapper_request,
    thesis_wrapper_summary_payload,
    write_thesis_wrapper_summary,
)

__all__ = [
    "_DEFAULT_EVAL_PRESET",
    "_DEFAULT_EVAL_PRESET_OVERRIDES",
    "_PRESET_PATHS",
    "_command_display",
    "_default_eval_preset_for_preset",
    "_resolve_cli_path",
    "_resolve_eval_stack_config",
    "_resolve_stack_config",
    "_run_step",
    "_summary_path",
    "ThesisWrapperCommands",
    "build_thesis_wrapper_parser",
    "ThesisWrapperPlan",
    "ThesisWrapperRequest",
    "ThesisWrapperResult",
    "ThesisWrapperInputs",
    "build_thesis_compare_command",
    "build_thesis_eval_command",
    "build_thesis_train_command",
    "build_thesis_wrapper_commands",
    "build_thesis_wrapper_commands_for_request",
    "build_thesis_wrapper_plan",
    "build_thesis_wrapper_plan_for_request",
    "main",
    "print_thesis_wrapper_presets",
    "run_thesis_wrapper_cli",
    "run_thesis_wrapper_commands",
    "run_thesis_wrapper_plan",
    "thesis_wrapper_repo_root",
    "thesis_wrapper_commands_for_plan",
    "thesis_wrapper_inputs_from_args",
    "thesis_wrapper_request",
    "thesis_wrapper_summary_payload",
    "write_thesis_wrapper_summary",
]


if __name__ == "__main__":
    main()
