from __future__ import annotations

from typing import Any

from weiss_rl.workflows.eval_support.eval_startup_dependencies import EvalStartupDependencies
from weiss_rl.workflows.eval_support.eval_startup_state import EvalStartup


def prepare_eval_startup(
    *,
    args: Any,
    run_label: str,
    dependencies: EvalStartupDependencies,
) -> EvalStartup:
    stack = dependencies.load_stack_config_fn(args.stack_config)
    config_hash256 = dependencies.compute_config_hash256_fn(stack)
    dependencies.require_matching_hash_fn(
        flag_name="--config-hash",
        expected=dependencies.expected_sha256_fn(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    spec_mismatch_policy = "hard_fail"
    contract = None
    if args.public_demo:
        public_demo_bundle = dependencies.public_demo_spec_bundle_fn()
        dependencies.assert_spec_bundle_contract_fn(args.spec_hash, public_demo_bundle)
        reported_spec_hash = dependencies.public_demo_spec_hash256_fn()
    else:
        contract = dependencies.load_verified_simulator_contract_fn(stack.root, expected_spec_hash=args.spec_hash)
        reported_spec_hash = contract.spec_hash256
    dependencies.print_startup_banner_fn(
        reported_spec_hash,
        config_hash256,
        run_label=run_label,
        spec_mismatch_policy=spec_mismatch_policy,
    )
    return EvalStartup(
        stack=stack,
        config_hash256=config_hash256,
        reported_spec_hash=reported_spec_hash,
        contract=contract,
    )
