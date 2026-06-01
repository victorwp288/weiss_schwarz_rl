from __future__ import annotations

from typing import Any

from weiss_rl.training.train_entrypoint_state import TrainManifestState, TrainStartupState


def prepare_train_manifest_state(*, args: Any, api: Any, startup: TrainStartupState) -> TrainManifestState:
    cli = startup.cli
    stack = cli.stack
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
        num_envs=cli.num_envs,
        unroll_length=cli.unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=int(startup.spec_bundle["action"]["pass_action_id"]),
        runtime_mode=cli.runtime_mode,
        learner_device=device,
    )
    policy_set_selection, policy_set_selection_details = api._resolve_policy_set_selection(
        stack,
        snapshot_registry_path=args.snapshot_registry_json,
        dev_eval_summaries_path=args.dev_eval_summaries_json,
    )
    manifest = api.RunManifest(
        run_id256=startup.run_id256,
        run_id64=startup.run_id64,
        start_nonce=startup.start_nonce,
        git_commit=startup.git_commit,
        git_dirty=api._git_dirty(),
        spec_hash256=startup.spec_hash256,
        config_hash256=startup.config_hash256,
        simulator=startup.simulator_info,
        spec_bundle=startup.spec_bundle,
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
    if cli.resume_run_dir is None:
        artifacts = api.write_run_artifacts(
            stack.root / "runs",
            manifest,
            run_label=cli.run_label or None,
        )
    else:
        artifacts = startup.resume_artifacts
        assert artifacts is not None

    run_summary_payload = api._load_json_object(artifacts.run_summary_path, label="run summary")
    api.augment_run_summary_payload(
        run_summary_payload,
        public_demo_enabled=cli.public_demo_enabled,
        runtime_mode=str(cli.runtime_mode),
        policy_set_selection_details=policy_set_selection_details,
        training_config=cli.training_config,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint_path=cli.init_from_checkpoint_path,
        resume_run_dir=cli.resume_run_dir,
        resume_checkpoint_path=cli.resume_checkpoint_path,
    )
    api._write_json(artifacts.run_summary_path, run_summary_payload)

    determinism_payload = api._load_json_object(artifacts.determinism_report_path, label="determinism report")
    api.augment_determinism_payload(
        determinism_payload,
        public_demo_enabled=cli.public_demo_enabled,
        runtime_mode=str(cli.runtime_mode),
        policy_set_selection_details=policy_set_selection_details,
        training_config=cli.training_config,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        seed_snapshot_run_dir=args.seed_snapshot_run_dir,
        init_from_checkpoint_path=cli.init_from_checkpoint_path,
        resume_checkpoint_path=cli.resume_checkpoint_path,
    )
    api._write_json(artifacts.determinism_report_path, determinism_payload)

    environment_payload = api._load_json_object(artifacts.environment_path, label="environment manifest")
    api.augment_environment_payload(
        environment_payload,
        root=stack.root,
        argv=api.sys.argv,
        hardware=manifest.hardware,
        init_from_checkpoint_path=cli.init_from_checkpoint_path,
        resume_checkpoint_path=cli.resume_checkpoint_path,
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
    if cli.resume_run_dir is None:
        print(f"Wrote manifest: {artifacts.manifest_path}")
    else:
        print(f"Resuming existing run directory: {artifacts.run_dir}")

    return TrainManifestState(
        artifacts=artifacts,
        manifest=manifest,
        device=device,
        profile=profile,
        seed=seed,
        policy_set_selection_details=policy_set_selection_details,
        tensorboard_logger=tensorboard_logger,
        run_summary_payload=run_summary_payload,
        determinism_payload=determinism_payload,
        environment_payload=environment_payload,
    )
