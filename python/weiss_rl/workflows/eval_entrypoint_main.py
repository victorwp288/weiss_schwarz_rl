from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_eval_entrypoint_main(
    *,
    build_eval_parser_fn: Callable[[], Any],
    validate_eval_args_fn: Callable[..., Any],
    prepare_eval_startup_fn: Callable[..., Any],
    run_eval_dispatch_fn: Callable[..., Any],
    eval_startup_dependencies_fn: Callable[[], Any],
    eval_dispatch_dependencies_fn: Callable[[], Any],
) -> None:
    parser = build_eval_parser_fn()
    args = parser.parse_args()
    validated = validate_eval_args_fn(parser=parser, args=args)
    startup = prepare_eval_startup_fn(
        args=args,
        run_label=validated.run_label,
        dependencies=eval_startup_dependencies_fn(),
    )
    run_eval_dispatch_fn(
        parser=parser,
        args=args,
        validated=validated,
        startup=startup,
        dependencies=eval_dispatch_dependencies_fn(),
    )


__all__ = ["run_eval_entrypoint_main"]
