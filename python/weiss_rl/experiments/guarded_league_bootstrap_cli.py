from __future__ import annotations

import argparse
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
    runtime_overrides_with_defaults,
)


def build_guarded_league_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded guided-league bootstrap controller")
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-snapshot-run-dir", type=Path, required=True)
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed B1/source run used for the canonical B1 anchor. "
            "Defaults to --seed-snapshot-run-dir for backwards compatibility."
        ),
    )
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment-updates", type=int, default=10)
    parser.add_argument(
        "--first-init-schedule-offset-updates",
        type=int,
        default=None,
        help=(
            "Optional guidance-schedule offset for the first --init-from-checkpoint segment. "
            "Use 0 to test a true bootstrap from imported weights instead of carrying source schedule time."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=288)
    parser.add_argument("--unroll-length", type=int, default=64)
    parser.add_argument("--runtime-mode", default="train_async_fast")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-interval-updates", type=int, default=5)
    parser.add_argument("--collection-backend", default="process")
    parser.add_argument("--override", action="append", default=None)
    parser.add_argument(
        "--use-stack-seed-snapshot-policy",
        action="store_true",
        help=(
            "Do not inject the guarded controller's default pinned seed-snapshot overrides. "
            "Use this for champion/hard-negative configs whose stack config intentionally imports all/source champions."
        ),
    )
    parser.add_argument("--no-profile-timers", action="store_true")
    parser.add_argument("--confirm-paired-seeds", type=int, default=64)
    parser.add_argument(
        "--publish-min-confirm-paired-seeds",
        type=int,
        default=256,
        help="Minimum confirm paired seeds required before publishing the selected alias.",
    )
    parser.add_argument(
        "--confirm-recent-candidate-count",
        type=int,
        default=1,
        help=(
            "Confirm this many most recent train snapshots after each segment before selecting. "
            "Use values >1 to let selected/best be the best confirmed checkpoint instead of the latest checkpoint."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--required-anchor", action="append", default=None)
    parser.add_argument("--confirm-opponent", action="append", default=None)
    parser.add_argument("--min-required-anchor-score", type=float, default=0.5)
    parser.add_argument("--reference-summary-json", type=Path, default=None)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--max-reference-drop", type=float, default=0.04)
    parser.add_argument(
        "--multiobjective-reference-summary-json",
        action="append",
        type=Path,
        default=None,
        help=(
            "Reference targeted-confirm summary for fixed/learned aggregate gates. "
            "Defaults to --reference-summary-json when omitted."
        ),
    )
    parser.add_argument("--multiobjective-fixed-opponent", action="append", default=None)
    parser.add_argument("--learned-guard-opponent", action="append", default=None)
    parser.add_argument("--min-multiobjective-fixed-score", type=float, default=0.5)
    parser.add_argument("--max-multiobjective-fixed-reference-drop", type=float, default=0.0)
    parser.add_argument("--min-learned-guard-score", type=float, default=0.5)
    parser.add_argument("--min-learned-guard-mean", type=float, default=0.5)
    parser.add_argument("--min-learned-guard-reference-delta", type=float, default=0.0)
    parser.add_argument("--max-learned-guard-reference-drop", type=float, default=None)
    parser.add_argument("--selected-alias-policy-id", default=DEFAULT_SELECTED_ALIAS_POLICY_ID)
    parser.add_argument(
        "--continue-unpublished-confirmed",
        action="store_true",
        help=(
            "Continue later segments from guard-passing selected checkpoints even when confirmation seed count is below "
            "--publish-min-confirm-paired-seeds. This never publishes the selected alias."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_guarded_league_bootstrap_args() -> argparse.Namespace:
    return build_guarded_league_bootstrap_parser().parse_args()


def uses_seed_snapshot_opponents(values: list[str] | None) -> bool:
    return any(str(value).strip().startswith("seed_") for value in values or [])


def has_seed_snapshot_pool_override(values: list[str] | None) -> bool:
    prefixes = (
        "league.pool.seed_snapshot_import_filter=",
        "league.pool.seed_snapshot_champion_import=",
    )
    return any(str(value).strip().startswith(prefixes) for value in values or [])


def validate_guarded_league_bootstrap_args(args: argparse.Namespace) -> None:
    opponent_args = [*(args.required_anchor or []), *(args.confirm_opponent or [])]
    if (
        uses_seed_snapshot_opponents(opponent_args)
        and not bool(args.use_stack_seed_snapshot_policy)
        and not has_seed_snapshot_pool_override(args.override)
    ):
        raise SystemExit(
            "seed_* learned opponents require --use-stack-seed-snapshot-policy or explicit "
            "league.pool.seed_snapshot_* overrides; the guarded controller's default pinned "
            "seed-snapshot policy will not import them"
        )
    if args.first_init_schedule_offset_updates is not None and int(args.first_init_schedule_offset_updates) < 0:
        raise SystemExit("--first-init-schedule-offset-updates must be >= 0")
    if int(args.confirm_recent_candidate_count) < 1:
        raise SystemExit("--confirm-recent-candidate-count must be >= 1")


def guarded_multiobjective_reference_summary_jsons(args: argparse.Namespace) -> tuple[Path, ...]:
    return tuple(
        args.multiobjective_reference_summary_json
        or ([args.reference_summary_json] if args.reference_summary_json is not None else [])
    )


def guarded_league_runtime_from_args(args: argparse.Namespace) -> LeagueSegmentRuntime:
    return LeagueSegmentRuntime(
        num_envs=int(args.num_envs),
        unroll_length=int(args.unroll_length),
        segment_updates=int(args.segment_updates),
        runtime_mode=str(args.runtime_mode),
        simulator_profile=str(args.profile),
        device=str(args.device),
        checkpoint_interval_updates=int(args.checkpoint_interval_updates),
        collection_backend=str(args.collection_backend),
        profile_timers=not bool(args.no_profile_timers),
        overrides=runtime_overrides_with_defaults(
            args.override,
            apply_seed_snapshot_defaults=not bool(args.use_stack_seed_snapshot_policy),
        ),
    )


def guarded_league_config_from_args(*, args: argparse.Namespace, repo_root: Path) -> GuardedLeagueBootstrapConfig:
    validate_guarded_league_bootstrap_args(args)
    return GuardedLeagueBootstrapConfig(
        repo_root=repo_root,
        init_checkpoint_path=Path(args.init_from_checkpoint),
        seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
        b1_baseline_run_dir=None if args.b1_baseline_run_dir is None else Path(args.b1_baseline_run_dir),
        run_prefix=str(args.run_prefix),
        stack_config=Path(args.stack_config),
        segments=int(args.segments),
        runtime=guarded_league_runtime_from_args(args),
        first_init_schedule_offset_updates=args.first_init_schedule_offset_updates,
        confirm_paired_seeds=int(args.confirm_paired_seeds),
        publish_min_confirm_paired_seeds=int(args.publish_min_confirm_paired_seeds),
        confirm_recent_candidate_count=int(args.confirm_recent_candidate_count),
        bootstrap_samples=int(args.bootstrap_samples),
        required_anchors=tuple(args.required_anchor or DEFAULT_CONFIRM_OPPONENTS),
        confirm_opponents=tuple(args.confirm_opponent or DEFAULT_CONFIRM_OPPONENTS),
        min_required_anchor_score=float(args.min_required_anchor_score),
        reference_anchor_scores=load_reference_scores_or_empty(args.reference_summary_json),
        multiobjective_reference_summary_jsons=guarded_multiobjective_reference_summary_jsons(args),
        multiobjective_fixed_opponents=tuple(args.multiobjective_fixed_opponent or FIXED_THESIS_OPPONENTS),
        learned_guard_opponents=tuple(args.learned_guard_opponent or ()),
        min_multiobjective_fixed_score=float(args.min_multiobjective_fixed_score),
        max_multiobjective_fixed_reference_drop=float(args.max_multiobjective_fixed_reference_drop),
        min_learned_guard_score=float(args.min_learned_guard_score),
        min_learned_guard_mean=float(args.min_learned_guard_mean),
        min_learned_guard_reference_delta=float(args.min_learned_guard_reference_delta),
        max_learned_guard_reference_drop=args.max_learned_guard_reference_drop,
        reference_label=str(args.reference_label),
        max_reference_drop=float(args.max_reference_drop),
        selected_alias_policy_id=str(args.selected_alias_policy_id),
        continue_unpublished_confirmed=bool(args.continue_unpublished_confirmed),
        dry_run=bool(args.dry_run),
    )


__all__ = [
    "build_guarded_league_bootstrap_parser",
    "guarded_league_config_from_args",
    "guarded_league_runtime_from_args",
    "guarded_multiobjective_reference_summary_jsons",
    "has_seed_snapshot_pool_override",
    "parse_guarded_league_bootstrap_args",
    "uses_seed_snapshot_opponents",
    "validate_guarded_league_bootstrap_args",
]
