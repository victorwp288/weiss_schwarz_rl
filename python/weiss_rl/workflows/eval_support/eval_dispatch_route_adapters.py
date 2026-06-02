from __future__ import annotations

import argparse
from typing import Any

from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
from weiss_rl.workflows.eval_support.eval_dispatch_request import EvalDispatchRequest, eval_dispatch_request
from weiss_rl.workflows.eval_support.eval_startup_state import EvalStartup, EvalValidatedArgs


def print_startup_verification_for_request(request: EvalDispatchRequest) -> None:
    startup = request.startup
    args = request.args
    dependencies = request.dependencies
    if startup.contract is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={startup.contract.simulator.get('compatibility_hash', '')} "
            f"sha256={startup.contract.spec_hash256}"
        )
    elif args.public_demo:
        print(
            "Verified public-demo spec bundle: "
            f"compat={dependencies.public_demo_spec_bundle_fn()['spec_hash']} sha256={startup.reported_spec_hash}"
        )


def _print_startup_verification(
    *,
    args: Any,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> None:
    print_startup_verification_for_request(
        eval_dispatch_request(
            parser=argparse.ArgumentParser(),
            args=args,
            validated=EvalValidatedArgs(
                run_label="", paired_seed_limit=None, stage1_paired_seeds=None, max_paired_seeds=None
            ),
            startup=startup,
            dependencies=dependencies,
        )
    )


def run_public_demo_eval_request(request: EvalDispatchRequest) -> None:
    request.dependencies.run_public_demo_eval_mode_fn(**request.public_demo_kwargs())


def run_public_demo_eval_route(
    *,
    args: Any,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> None:
    run_public_demo_eval_request(
        eval_dispatch_request(
            parser=argparse.ArgumentParser(),
            args=args,
            validated=EvalValidatedArgs(
                run_label="", paired_seed_limit=None, stage1_paired_seeds=None, max_paired_seeds=None
            ),
            startup=startup,
            dependencies=dependencies,
        )
    )


def run_canonical_eval_request(request: EvalDispatchRequest) -> int:
    return int(request.dependencies.run_canonical_eval_pipeline_fn(**request.canonical_kwargs()))


def run_canonical_eval_route(
    *,
    parser: argparse.ArgumentParser,
    args: Any,
    validated: EvalValidatedArgs,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> int:
    return run_canonical_eval_request(
        eval_dispatch_request(
            parser=parser,
            args=args,
            validated=validated,
            startup=startup,
            dependencies=dependencies,
        )
    )


def run_summary_only_eval_request(request: EvalDispatchRequest) -> None:
    request.dependencies.run_summary_only_eval_mode_fn(**request.summary_only_kwargs())


def run_summary_only_eval_route(
    *,
    args: Any,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> None:
    run_summary_only_eval_request(
        eval_dispatch_request(
            parser=argparse.ArgumentParser(),
            args=args,
            validated=EvalValidatedArgs(
                run_label="", paired_seed_limit=None, stage1_paired_seeds=None, max_paired_seeds=None
            ),
            startup=startup,
            dependencies=dependencies,
        )
    )


__all__ = [
    "_print_startup_verification",
    "print_startup_verification_for_request",
    "run_canonical_eval_route",
    "run_canonical_eval_request",
    "run_public_demo_eval_route",
    "run_public_demo_eval_request",
    "run_summary_only_eval_route",
    "run_summary_only_eval_request",
]
