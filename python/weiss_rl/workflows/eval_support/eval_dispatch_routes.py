from __future__ import annotations

import argparse
from typing import Any

from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
from weiss_rl.workflows.eval_support.eval_dispatch_request import eval_dispatch_request
from weiss_rl.workflows.eval_support.eval_dispatch_route_adapters import (
    _print_startup_verification,
    print_startup_verification_for_request,
    run_canonical_eval_request,
    run_public_demo_eval_request,
    run_summary_only_eval_request,
)
from weiss_rl.workflows.eval_support.eval_startup_state import EvalStartup, EvalValidatedArgs


def run_eval_dispatch(
    *,
    parser: argparse.ArgumentParser,
    args: Any,
    validated: EvalValidatedArgs,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> None:
    request = eval_dispatch_request(
        parser=parser,
        args=args,
        validated=validated,
        startup=startup,
        dependencies=dependencies,
    )
    print_startup_verification_for_request(request)

    if request.is_public_demo:
        run_public_demo_eval_request(request)
        return

    if request.has_run_dir:
        raise SystemExit(run_canonical_eval_request(request))

    if not request.has_episodes_jsonl:
        print(
            "Evaluation contract check complete; no episodes were summarized. "
            f"Seed sets: {sorted(startup.stack.seed_sets)}"
        )
        return

    run_summary_only_eval_request(request)


__all__ = ["_print_startup_verification", "run_eval_dispatch"]
