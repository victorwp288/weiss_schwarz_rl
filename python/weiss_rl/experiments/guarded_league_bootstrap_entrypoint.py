from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.guarded_league_bootstrap import (
    DEFAULT_CONFIRM_OPPONENTS,
    DEFAULT_RUN_PREFIX,
    DEFAULT_SELECTED_ALIAS_POLICY_ID,
    DEFAULT_STACK_CONFIG,
    FIXED_THESIS_OPPONENTS,
    GuardedLeagueBootstrapConfig,
    LeagueSegmentRuntime,
    load_reference_scores_or_empty,
    run_guarded_league_bootstrap,
    runtime_overrides_with_defaults,
)
from weiss_rl.experiments.guarded_league_bootstrap_cli import (
    build_guarded_league_bootstrap_parser,
    guarded_league_config_from_args,
    guarded_league_runtime_from_args,
    guarded_multiobjective_reference_summary_jsons,
    parse_guarded_league_bootstrap_args,
)
from weiss_rl.experiments.guarded_league_bootstrap_cli import (
    has_seed_snapshot_pool_override as _has_seed_snapshot_pool_override,
)
from weiss_rl.experiments.guarded_league_bootstrap_cli import (
    uses_seed_snapshot_opponents as _uses_seed_snapshot_opponents,
)
from weiss_rl.experiments.guarded_league_bootstrap_cli import (
    validate_guarded_league_bootstrap_args as _validate_seed_snapshot_policy,
)

parse_args = parse_guarded_league_bootstrap_args

__all__ = [
    "DEFAULT_CONFIRM_OPPONENTS",
    "DEFAULT_RUN_PREFIX",
    "DEFAULT_SELECTED_ALIAS_POLICY_ID",
    "DEFAULT_STACK_CONFIG",
    "FIXED_THESIS_OPPONENTS",
    "GuardedLeagueBootstrapConfig",
    "LeagueSegmentRuntime",
    "_has_seed_snapshot_pool_override",
    "_uses_seed_snapshot_opponents",
    "_validate_seed_snapshot_policy",
    "build_guarded_league_bootstrap_parser",
    "guarded_league_config_from_args",
    "guarded_league_runtime_from_args",
    "guarded_multiobjective_reference_summary_jsons",
    "load_reference_scores_or_empty",
    "main",
    "parse_args",
    "parse_guarded_league_bootstrap_args",
    "run_guarded_league_bootstrap",
    "runtime_overrides_with_defaults",
]


def main() -> None:
    config = guarded_league_config_from_args(args=parse_args(), repo_root=Path.cwd())
    summary = run_guarded_league_bootstrap(config)
    print(json.dumps({"status": summary["status"], "summary_path": summary["summary_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
