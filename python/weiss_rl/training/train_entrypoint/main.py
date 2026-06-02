"""CLI main body for the package-owned training entrypoint."""

from __future__ import annotations

from typing import Any

from weiss_rl.training.train_entrypoint.phases import (
    TrainCliState,
    TrainManifestState,
    TrainStartupState,
    execute_train_run,
    prepare_train_manifest_state,
    prepare_train_startup_state,
    require_explicit_resume_geometry,
    resolve_train_cli_state,
)

__all__ = [
    "TrainCliState",
    "TrainManifestState",
    "TrainStartupState",
    "_require_explicit_resume_geometry",
    "execute_train_run",
    "prepare_train_manifest_state",
    "prepare_train_startup_state",
    "resolve_train_cli_state",
    "run_train_main",
]


def _require_explicit_resume_geometry(parser: Any, args: Any) -> None:
    require_explicit_resume_geometry(parser, args)


def run_train_main(api: Any) -> None:
    """Run the canonical training CLI through compatibility hook modules."""

    parser = api.build_train_parser()
    args = parser.parse_args()
    cli = resolve_train_cli_state(parser=parser, args=args, api=api)
    startup = prepare_train_startup_state(parser=parser, args=args, api=api, cli=cli)
    manifest_state = prepare_train_manifest_state(args=args, api=api, startup=startup)
    tensorboard_logger = manifest_state.tensorboard_logger

    try:
        execute_train_run(args=args, api=api, startup=startup, manifest_state=manifest_state)
    finally:
        tensorboard_logger.close()
