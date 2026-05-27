"""CLI main body for the path-based training entrypoint."""

from __future__ import annotations

from typing import Any, cast


def _require_explicit_resume_geometry(parser: Any, args: Any) -> None:
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


def run_train_main(api: Any) -> None:
    """Run `python/scripts/train.py` through script-level compatibility hooks."""

    parser = api.build_train_parser()
    args = parser.parse_args()
    run_label = api._resolve_run_label(parser, args.run_label, args.run_id_alias)

    _require_explicit_resume_geometry(parser, args)

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
    if public_demo_enabled:
        public_demo_bundle = api.public_demo_spec_bundle()
        api.assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        spec_bundle = public_demo_bundle
        spec_hash256 = api.public_demo_spec_hash256()
        simulator_info = api.public_demo_simulator_info()
    else:
        simulator_contract = api.load_verified_simulator_contract(stack.root, expected_spec_hash=args.spec_hash)
        spec_bundle = simulator_contract.spec_bundle
        spec_hash256 = simulator_contract.spec_hash256
        simulator_info = simulator_contract.simulator
    config_hash256 = api.compute_config_hash256(stack)
    api._require_matching_hash(
        flag_name="--config-hash",
        expected=api._expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = api._git_commit()
    start_nonce = api._start_nonce()
    if resume_run_dir is None:
        run_identity = api.new_run_identity(
            spec_hash256=spec_hash256,
            config_hash256=config_hash256,
            git_commit=git_commit,
            start_nonce=start_nonce,
            run_label=run_label,
        )
        run_id256 = run_identity.run_id256
        run_id64 = run_identity.run_id64
        run_dir_name = run_identity.run_dir_name
    else:
        artifacts = api._run_artifacts_from_existing_run_dir(resume_run_dir)
        run_identity = api.resume_run_identity(
            api._load_json_object(artifacts.manifest_path, label="resume manifest"),
            manifest_path=artifacts.manifest_path,
            run_dir_name=artifacts.run_dir_name,
            expected_spec_hash256=spec_hash256,
            expected_config_hash256=config_hash256,
        )
        run_id256 = run_identity.run_id256
        run_id64 = run_identity.run_id64
        run_dir_name = run_identity.run_dir_name

    api.print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label or ("" if resume_run_dir is None else run_dir_name),
        run_dir_name=run_dir_name,
        spec_mismatch_policy=api._spec_mismatch_policy(stack),
    )
    spec_bundle_message = (
        "Loaded synthetic public-demo spec bundle: " if public_demo_enabled else "Verified runtime spec bundle: "
    )
    print(spec_bundle_message + f"compat={simulator_info.get('compatibility_hash', '')} sha256={spec_hash256}")
    print(f"Loaded stack config with {len(stack.components)} components")

    device = api._resolve_device(stack, args.device)
    profile = api._resolve_runtime_profile(stack, "" if args.profile is None else args.profile)
    seed = api._resolve_seed(stack, args.seed)
    reproducibility_config = stack.config.reproducibility
    seed_derivation_config = None if reproducibility_config is None else reproducibility_config.seed_derivation
    seed_derivation_payload = {
        "config_base_seed64": None if seed_derivation_config is None else int(seed_derivation_config.base_seed64),
        "effective_base_seed64": int(seed),
        "cli_seed_override": args.seed is not None,
        "actor_seed_formula": (
            "hash64(base_seed64, actor_id)"
            if seed_derivation_config is None
            else str(seed_derivation_config.actor_seed_formula)
        ),
        "episode_seed_formula": (
            "hash64(actor_seed64, env_id, episode_index)"
            if seed_derivation_config is None
            else str(seed_derivation_config.episode_seed_formula)
        ),
    }
    actor_device_layout = api._manifest_actor_device_layout(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=int(spec_bundle["action"]["pass_action_id"]),
        runtime_mode=runtime_mode,
        learner_device=device,
    )
    policy_set_selection, policy_set_selection_details = api._resolve_policy_set_selection(
        stack,
        snapshot_registry_path=args.snapshot_registry_json,
        dev_eval_summaries_path=args.dev_eval_summaries_json,
    )
    manifest = api.RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=api._git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_info,
        spec_bundle=spec_bundle,
        config_canonical=api.canonical_config_dict(stack),
        seed_derivation=seed_derivation_payload,
        seed_files=api.build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=api._hardware_summary(
            device,
            actor_device=("cpu" if stack.config.system is None else stack.config.system.actor_device),
            actor_device_layout=actor_device_layout,
        ),
        evaluation_pinning=api._evaluation_pinning(stack),
        policy_set_selection=policy_set_selection,
        policy_set_selection_details=policy_set_selection_details,
    )
    if resume_run_dir is None:
        artifacts = api.write_run_artifacts(
            stack.root / "runs",
            manifest,
            run_label=run_label or None,
        )
    else:
        artifacts = api._run_artifacts_from_existing_run_dir(resume_run_dir)
    run_summary_payload = api._load_json_object(artifacts.run_summary_path, label="run summary")
    api.augment_run_summary_payload(
        run_summary_payload,
        public_demo_enabled=public_demo_enabled,
        runtime_mode=str(runtime_mode),
        policy_set_selection_details=policy_set_selection_details,
        training_config=training_config,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint_path=init_from_checkpoint_path,
        resume_run_dir=resume_run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    api._write_json(artifacts.run_summary_path, run_summary_payload)

    determinism_payload = api._load_json_object(artifacts.determinism_report_path, label="determinism report")
    api.augment_determinism_payload(
        determinism_payload,
        public_demo_enabled=public_demo_enabled,
        runtime_mode=str(runtime_mode),
        policy_set_selection_details=policy_set_selection_details,
        training_config=training_config,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint_path=init_from_checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    api._write_json(artifacts.determinism_report_path, determinism_payload)

    environment_payload = api._load_json_object(artifacts.environment_path, label="environment manifest")
    api.augment_environment_payload(
        environment_payload,
        root=stack.root,
        argv=api.sys.argv,
        hardware=manifest.hardware,
        init_from_checkpoint_path=init_from_checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    api._write_json(artifacts.environment_path, environment_payload)
    tensorboard_logger = api.TensorBoardLogger(artifacts.layout.tensorboard_dir)
    if not tensorboard_logger.enabled:
        unavailable_reason = api.tensorboard_unavailable_reason()
        print(
            "TensorBoard logging is disabled: "
            + ("SummaryWriter unavailable" if unavailable_reason is None else unavailable_reason),
            file=api.sys.stderr,
        )
    else:
        tensorboard_logger.log_run_context(
            manifest=manifest.to_dict(),
            environment=environment_payload,
            run_summary=run_summary_payload,
            determinism_report=determinism_payload,
        )
    if resume_run_dir is None:
        print(f"Wrote manifest: {artifacts.manifest_path}")
    else:
        print(f"Resuming existing run directory: {artifacts.run_dir}")

    try:
        if public_demo_enabled:
            staged = api.stage_public_demo_run(artifacts.run_dir)
            print(
                "Staged public-demo toy catalog and policy bundle: "
                f"mode={api.PUBLIC_DEMO_MODE} policy_count={len(staged.policy_ids)} "
                f"catalog={staged.catalog_path}"
            )
            print(
                "Public demo mode is intentionally synthetic and demo-only. "
                "It does not execute simulator training or claim thesis-grade results."
            )
            return

        if manifest_only_reason is not None:
            api._print_manifest_only_message(manifest_only_reason)
            return

        runtime_prerequisite_failure = api._runtime_training_prerequisite_failure(stack)
        if runtime_prerequisite_failure is not None:
            api._raise_runtime_prerequisite_failure(runtime_prerequisite_failure)
        noleague_prerequisite_failure = api._noleague_training_prerequisite_failure(stack)
        if noleague_prerequisite_failure is not None:
            api._raise_noleague_training_prerequisite_failure(noleague_prerequisite_failure)

        assert training_config is not None
        execution_settings = api.resolve_training_execution_settings(
            training_config=training_config,
            checkpoint_interval_override=args.checkpoint_interval_updates,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            seed_snapshot_run_dir=args.seed_snapshot_run_dir,
            init_from_checkpoint=args.init_from_checkpoint,
        )

        profiling_message = api.profiling_enabled_message(training_config)
        if profiling_message is not None:
            print(profiling_message)

        metrics = api._run_minimal_training(
            stack=stack,
            contract=simulator_contract,
            artifacts=artifacts,
            num_envs=num_envs,
            unroll_length=unroll_length,
            max_updates=max_updates,
            profile=profile,
            device=device,
            seed=seed,
            checkpoint_interval_updates=execution_settings.checkpoint_interval_updates,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            runtime_mode=runtime_mode,
            b1_baseline_run_dir=execution_settings.b1_baseline_run_dir,
            seed_snapshot_run_dir=execution_settings.seed_snapshot_run_dir,
            profile_timers=execution_settings.profile_timers,
            torch_profiler=execution_settings.torch_profiler,
            resume_checkpoint_path=resume_checkpoint_path,
            init_from_checkpoint_path=execution_settings.init_from_checkpoint_path,
            init_schedule_offset_override_updates=init_schedule_offset_override_updates,
            tensorboard_logger=tensorboard_logger,
        )
        print(
            "Completed canonical single-node training run: "
            f"loss={metrics.get('loss', 0.0):.6f} "
            f"policy_loss={metrics.get('policy_loss', 0.0):.6f} "
            f"value_loss={metrics.get('value_loss', 0.0):.6f} "
            f"entropy={metrics.get('entropy', 0.0):.6f}"
        )
    finally:
        tensorboard_logger.close()
