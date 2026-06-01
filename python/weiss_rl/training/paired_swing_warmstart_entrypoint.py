"""Entrypoint facade for paired-swing warmstart runs."""

from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.training.paired_swing_warmstart_cli import (
    build_paired_swing_warmstart_parser,
    parse_paired_swing_warmstart_args,
    validate_paired_swing_warmstart_args,
)
from weiss_rl.training.paired_swing_warmstart_runtime import (
    _publish_paired_swing_snapshot,
    _sha256_file,
    _write_run_contract_artifacts,
    run_paired_swing_warmstart,
)
from weiss_rl.training.warmstart_replay_support import (
    _initial_hidden_state,
    _opponent_context_indices_for_episodes,
    _sample_episode_indices,
    _source_opponent_policy_ids_by_episode,
)

_build_parser = build_paired_swing_warmstart_parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_paired_swing_warmstart(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_build_parser",
    "_initial_hidden_state",
    "_opponent_context_indices_for_episodes",
    "_publish_paired_swing_snapshot",
    "_sample_episode_indices",
    "_sha256_file",
    "_source_opponent_policy_ids_by_episode",
    "_write_run_contract_artifacts",
    "main",
    "parse_paired_swing_warmstart_args",
    "run_paired_swing_warmstart",
    "validate_paired_swing_warmstart_args",
]
