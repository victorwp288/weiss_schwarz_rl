from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.experiments import trajectory_policy_drift_scoring as _scoring
from weiss_rl.experiments.trajectory_policy_drift_cli import (
    build_trajectory_policy_drift_parser,
)
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    PolicyScores,
    PolicySpec,
    build_trajectory_policy_drift_report,
    parse_policy_specs,
    print_trajectory_policy_drift_summary,
    source_opponent_policy_ids_by_episode,
    write_trajectory_policy_drift_report,
)
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    parse_policy_spec as _parse_policy_spec,
)
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    trajectory_row_coordinates as _row_coordinates,
)
from weiss_rl.experiments.trajectory_policy_drift_reporting import (
    trajectory_row_group_labels as _row_group_labels,
)
from weiss_rl.experiments.trajectory_policy_drift_runtime import run_trajectory_policy_drift

_build_parser = build_trajectory_policy_drift_parser
_source_opponent_policy_ids_by_episode = source_opponent_policy_ids_by_episode
_print_summary = print_trajectory_policy_drift_summary
_configure_torch_determinism = _scoring.configure_torch_determinism
_action_catalog_from_stack_spec = _scoring.action_catalog_from_stack_spec
_family_metadata = _scoring.family_metadata
_score_policy = _scoring.score_policy
_dense_policy_scores_from_packed_logits = _scoring.dense_policy_scores_from_packed_logits
_safe_actions_for_scoring = _scoring.safe_actions_for_scoring

__all__ = [
    "PolicyScores",
    "PolicySpec",
    "_action_catalog_from_stack_spec",
    "_build_parser",
    "_configure_torch_determinism",
    "_dense_policy_scores_from_packed_logits",
    "_family_metadata",
    "_parse_policy_spec",
    "_print_summary",
    "_row_coordinates",
    "_row_group_labels",
    "_safe_actions_for_scoring",
    "_score_policy",
    "_source_opponent_policy_ids_by_episode",
    "build_trajectory_policy_drift_report",
    "main",
    "parse_policy_specs",
    "print_trajectory_policy_drift_summary",
    "run_trajectory_policy_drift",
    "source_opponent_policy_ids_by_episode",
    "write_trajectory_policy_drift_report",
]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_trajectory_policy_drift_parser().parse_args(argv)
    run_trajectory_policy_drift(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
