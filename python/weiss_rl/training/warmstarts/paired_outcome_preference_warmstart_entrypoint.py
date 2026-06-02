"""Entrypoint facade for paired-outcome preference warmstart runs."""

from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_cli import (
    build_paired_outcome_preference_warmstart_parser,
    parse_paired_outcome_preference_warmstart_args,
    validate_paired_outcome_preference_warmstart_args,
)
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_runtime import (
    _publish_preference_snapshot,
    _sha256_file,
    _write_run_contract_artifacts,
    run_paired_outcome_preference_warmstart,
)
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_support import (
    _initial_hidden_state,
    _opponent_context_indices_for_episodes,
    _parse_pair_role_selectors,
    _parse_pair_weights,
    _preference_group_indices_for_episodes,
    _preference_pair_role_mask,
    _preference_pair_weight_matrix,
    _scale_optimizer_learning_rates,
    _serialize_pair_role_selectors,
    _source_opponent_policy_ids_by_episode,
)

_build_parser = build_paired_outcome_preference_warmstart_parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_paired_outcome_preference_warmstart(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_build_parser",
    "_initial_hidden_state",
    "_opponent_context_indices_for_episodes",
    "_parse_pair_role_selectors",
    "_parse_pair_weights",
    "_preference_group_indices_for_episodes",
    "_preference_pair_role_mask",
    "_preference_pair_weight_matrix",
    "_publish_preference_snapshot",
    "_scale_optimizer_learning_rates",
    "_serialize_pair_role_selectors",
    "_sha256_file",
    "_source_opponent_policy_ids_by_episode",
    "_write_run_contract_artifacts",
    "main",
    "parse_paired_outcome_preference_warmstart_args",
    "run_paired_outcome_preference_warmstart",
    "validate_paired_outcome_preference_warmstart_args",
]
