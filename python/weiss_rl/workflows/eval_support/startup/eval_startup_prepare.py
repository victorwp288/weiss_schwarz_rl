from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.workflows.eval_support.startup.eval_startup_dependencies import EvalStartupDependencies
from weiss_rl.workflows.eval_support.startup.eval_startup_state import EvalStartup


@dataclass(frozen=True, slots=True)
class EvalStartupRouteStep:
    step_id: str
    title: str
    route: str
    output: str

    def as_payload(self) -> dict[str, str]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "route": self.route,
            "output": self.output,
        }


EVAL_STARTUP_ROUTE: tuple[EvalStartupRouteStep, ...] = (
    EvalStartupRouteStep(
        step_id="load_stack",
        title="Load stack",
        route="Read the requested evaluation stack config before resolving policy inputs.",
        output="stack",
    ),
    EvalStartupRouteStep(
        step_id="verify_config_identity",
        title="Verify config identity",
        route="Compute the canonical stack hash and compare it with --config-hash when supplied.",
        output="config_hash256",
    ),
    EvalStartupRouteStep(
        step_id="select_spec_source",
        title="Select spec source",
        route="Use the synthetic public-demo bundle for demo runs; otherwise load and verify the runtime simulator.",
        output="reported_spec_hash, contract",
    ),
    EvalStartupRouteStep(
        step_id="announce_startup",
        title="Announce startup",
        route="Print the startup banner with hard-fail spec mismatch policy before dispatch.",
        output="operator-facing startup summary",
    ),
)


def eval_startup_route_payload() -> list[dict[str, str]]:
    return [step.as_payload() for step in EVAL_STARTUP_ROUTE]


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


__all__ = ["EVAL_STARTUP_ROUTE", "EvalStartupRouteStep", "eval_startup_route_payload", "prepare_eval_startup"]
