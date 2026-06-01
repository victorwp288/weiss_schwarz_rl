from __future__ import annotations

from typing import Any, cast

from weiss_rl.training.train_entrypoint_state import TrainCliState


def require_explicit_resume_geometry(parser: Any, args: Any) -> None:
    resume_requested = args.resume_run_dir is not None or bool(str(args.resume_from).strip())
    if not resume_requested:
        return
    missing_resume_geometry = []
    if args.num_envs is None:
        missing_resume_geometry.append("--num-envs")
    if args.unroll_length is None:
        missing_resume_geometry.append("--unroll-length")
    if args.runtime_mode is None:
        missing_resume_geometry.append("--runtime-mode")
    if args.profile is None:
        missing_resume_geometry.append("--profile")
    if missing_resume_geometry:
        parser.error(
            "resume requires explicit runtime geometry to avoid silent batch-size/profile changes: "
            + ", ".join(missing_resume_geometry)
        )


def resolve_train_cli_state(*, parser: Any, args: Any, api: Any) -> TrainCliState:
    run_label = api._resolve_run_label(parser, args.run_label, args.run_id_alias)
    require_explicit_resume_geometry(parser, args)

    num_envs = api._require_positive_int("--num-envs", 2 if args.num_envs is None else args.num_envs)
    unroll_length = api._require_positive_int(
        "--unroll-length",
        4 if args.unroll_length is None else args.unroll_length,
    )
    max_updates = api._require_positive_int("--max-updates", args.max_updates)
    runtime_mode = cast(api.QueueRuntimeMode, args.runtime_mode or "train_ordered")

    stack = api.load_stack_config(args.stack_config)
    stack = api.apply_stack_overrides(stack, api.parse_override_tokens(args.config_override))
    stack = api._apply_training_flag_overrides(
        stack,
        enable_profile_timers=bool(args.profile_timers),
        enable_torch_profiler=bool(args.torch_profiler),
    )
    training_config = stack.config.training
    manifest_only_reason = api._manifest_scaffold_only_reason(stack)
    if training_config is None and manifest_only_reason is None:
        parser.error("stack config is missing training")

    public_demo_enabled = bool(args.public_demo)
    resume_run_dir = None if args.resume_run_dir is None else args.resume_run_dir.resolve()
    resume_checkpoint_path = api._resolve_resume_checkpoint_path(
        resume_from=str(args.resume_from),
        resume_run_dir=resume_run_dir,
    )
    init_from_checkpoint_path = None if args.init_from_checkpoint is None else args.init_from_checkpoint.resolve()
    init_schedule_offset_override_updates = args.init_schedule_offset_updates
    if init_from_checkpoint_path is not None and (resume_run_dir is not None or resume_checkpoint_path is not None):
        parser.error("--init-from-checkpoint starts a fresh run and cannot be combined with checkpoint resume")
    if init_schedule_offset_override_updates is not None:
        if init_from_checkpoint_path is None:
            parser.error("--init-schedule-offset-updates requires --init-from-checkpoint")
        if int(init_schedule_offset_override_updates) < 0:
            parser.error("--init-schedule-offset-updates must be >= 0")
    if public_demo_enabled and (
        resume_run_dir is not None or resume_checkpoint_path is not None or init_from_checkpoint_path is not None
    ):
        parser.error("Public demo mode does not support checkpoint resume or checkpoint initialization")

    return TrainCliState(
        run_label=run_label,
        num_envs=num_envs,
        unroll_length=unroll_length,
        max_updates=max_updates,
        runtime_mode=runtime_mode,
        stack=stack,
        training_config=training_config,
        manifest_only_reason=manifest_only_reason,
        public_demo_enabled=public_demo_enabled,
        resume_run_dir=resume_run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        init_from_checkpoint_path=init_from_checkpoint_path,
        init_schedule_offset_override_updates=init_schedule_offset_override_updates,
    )
