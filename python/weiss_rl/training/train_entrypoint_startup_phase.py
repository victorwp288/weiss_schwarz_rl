from __future__ import annotations

from typing import Any

from weiss_rl.training.train_entrypoint_state import TrainCliState, TrainStartupState


def prepare_train_startup_state(*, parser: Any, args: Any, api: Any, cli: TrainCliState) -> TrainStartupState:
    simulator_contract = None
    if cli.public_demo_enabled:
        public_demo_bundle = api.public_demo_spec_bundle()
        api.assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        spec_bundle = public_demo_bundle
        spec_hash256 = api.public_demo_spec_hash256()
        simulator_info = api.public_demo_simulator_info()
    else:
        simulator_contract = api.load_verified_simulator_contract(cli.stack.root, expected_spec_hash=args.spec_hash)
        spec_bundle = simulator_contract.spec_bundle
        spec_hash256 = simulator_contract.spec_hash256
        simulator_info = simulator_contract.simulator

    config_hash256 = api.compute_config_hash256(cli.stack)
    api._require_matching_hash(
        flag_name="--config-hash",
        expected=api._expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    git_commit = api._git_commit()
    start_nonce = api._start_nonce()
    resume_artifacts = None
    if cli.resume_run_dir is None:
        run_identity = api.new_run_identity(
            spec_hash256=spec_hash256,
            config_hash256=config_hash256,
            git_commit=git_commit,
            start_nonce=start_nonce,
            run_label=cli.run_label,
        )
    else:
        resume_artifacts = api._run_artifacts_from_existing_run_dir(cli.resume_run_dir)
        run_identity = api.resume_run_identity(
            api._load_json_object(resume_artifacts.manifest_path, label="resume manifest"),
            manifest_path=resume_artifacts.manifest_path,
            run_dir_name=resume_artifacts.run_dir_name,
            expected_spec_hash256=spec_hash256,
            expected_config_hash256=config_hash256,
        )

    api.print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_identity.run_id64,
        run_id256=run_identity.run_id256,
        run_label=cli.run_label or ("" if cli.resume_run_dir is None else run_identity.run_dir_name),
        run_dir_name=run_identity.run_dir_name,
        spec_mismatch_policy=api._spec_mismatch_policy(cli.stack),
    )
    spec_bundle_message = (
        "Loaded synthetic public-demo spec bundle: " if cli.public_demo_enabled else "Verified runtime spec bundle: "
    )
    print(spec_bundle_message + f"compat={simulator_info.get('compatibility_hash', '')} sha256={spec_hash256}")
    print(f"Loaded stack config with {len(cli.stack.components)} components")

    return TrainStartupState(
        cli=cli,
        simulator_contract=simulator_contract,
        spec_bundle=spec_bundle,
        spec_hash256=spec_hash256,
        simulator_info=simulator_info,
        config_hash256=config_hash256,
        git_commit=git_commit,
        start_nonce=start_nonce,
        run_id256=run_identity.run_id256,
        run_id64=run_identity.run_id64,
        run_dir_name=run_identity.run_dir_name,
        resume_artifacts=resume_artifacts,
    )
