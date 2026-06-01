"""Deterministic subprocess command builders for package workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from weiss_rl.workflows.profiles import EVAL_STACK_CONFIG, MAIN_STACK_CONFIG, TrainProfile

__all__ = [
    "DEFAULT_GUARD_REQUIRED_ANCHORS",
    "build_b2_audit_command",
    "build_eval_command",
    "build_figures_command",
    "build_guard_run_command",
    "build_guarded_league_bootstrap_command",
    "build_guided_bootstrap_loop_command",
    "build_train_command",
]

DEFAULT_GUARD_REQUIRED_ANCHORS = (
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)


def build_train_command(
    *,
    python_exe: str,
    stack_config: Path,
    run_label: str,
    profile: TrainProfile,
    b1_baseline_run_dir: Path | None = None,
    seed_snapshot_run_dir: Path | None = None,
    init_from_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        python_exe,
        "python/scripts/train.py",
        "--stack-config",
        stack_config.as_posix(),
        "--run-label",
        run_label,
        "--num-envs",
        str(profile.num_envs),
        "--unroll-length",
        str(profile.unroll_length),
        "--max-updates",
        str(profile.max_updates),
        "--runtime-mode",
        profile.runtime_mode,
        "--profile",
        profile.simulator_profile,
        "--device",
        profile.device,
    ]
    for override in profile.overrides:
        command.extend(["--override", override])
    if profile.checkpoint_interval_updates is not None:
        command.extend(["--checkpoint-interval-updates", str(profile.checkpoint_interval_updates)])
    if b1_baseline_run_dir is not None:
        command.extend(["--b1-baseline-run-dir", b1_baseline_run_dir.as_posix()])
    if seed_snapshot_run_dir is not None:
        command.extend(["--seed-snapshot-run-dir", seed_snapshot_run_dir.as_posix()])
    if init_from_checkpoint is not None:
        command.extend(["--init-from-checkpoint", init_from_checkpoint.as_posix()])
    return command


def build_eval_command(
    *,
    python_exe: str,
    run_dir: Path,
    b1_baseline_run_dir: Path | None,
    smoke: bool,
) -> list[str]:
    stack_config = MAIN_STACK_CONFIG if smoke else EVAL_STACK_CONFIG
    command = [
        python_exe,
        "python/scripts/eval.py",
        "--stack-config",
        stack_config.as_posix(),
        "--run-dir",
        run_dir.as_posix(),
    ]
    if b1_baseline_run_dir is not None:
        command.extend(["--b1-baseline-run-dir", b1_baseline_run_dir.as_posix()])
    if smoke:
        for policy_id in (
            "B0 RandomLegal",
            "B1 NoLeague baseline",
            "B2 HeuristicPublic",
            "B3 HeuristicPublicAggro",
            "B4 HeuristicPublicControl",
        ):
            command.extend(["--policy-id", policy_id])
        command.extend(
            [
                "--paired-seed-limit",
                "1",
                "--stage1-paired-seeds",
                "1",
                "--max-paired-seeds",
                "1",
                "--bootstrap-samples",
                "16",
                "--skip-metagame",
                "--skip-figures",
                "--skip-readiness",
            ]
        )
    return command


def build_figures_command(
    *,
    python_exe: str,
    run_dir: Path,
    fig_id: str,
    formats: Sequence[str],
) -> list[str]:
    command = [python_exe, "python/scripts/make_figures.py", "--run-dir", run_dir.as_posix()]
    resolved_fig_id = str(fig_id).strip()
    if resolved_fig_id:
        command.extend(["--fig-id", resolved_fig_id])
    for fmt in formats:
        command.extend(["--format", str(fmt)])
    return command


def build_b2_audit_command(
    *,
    python_exe: str,
    run_dir: Path,
    episodes_jsonl: Path,
    policy_id: str,
    output_run_dir: Path | None,
    snapshot_registry_json: Path | None,
    summary_json: Path | None,
    top_k: int,
    top_actions: int,
    allow_policy_id_mismatch: bool,
    accepted_snapshot_config_hashes: Sequence[str],
) -> list[str]:
    resolved_output_run_dir = output_run_dir or (run_dir / "eval" / "b2_disagreement")
    command = [
        python_exe,
        "python/scripts/b2_disagreement_audit.py",
        "--stack-config",
        EVAL_STACK_CONFIG.as_posix(),
        "--run-dir",
        run_dir.as_posix(),
        "--output-run-dir",
        resolved_output_run_dir.as_posix(),
        "--episodes-jsonl",
        episodes_jsonl.as_posix(),
        "--policy-id",
        policy_id,
        "--top-k",
        str(top_k),
        "--top-actions",
        str(top_actions),
    ]
    if allow_policy_id_mismatch:
        command.append("--allow-policy-id-mismatch")
    for config_hash in accepted_snapshot_config_hashes:
        command.extend(["--accept-snapshot-config-hash", str(config_hash)])
    if snapshot_registry_json is not None:
        command.extend(["--snapshot-registry-json", snapshot_registry_json.as_posix()])
    if summary_json is not None:
        command.extend(["--summary-json", summary_json.as_posix()])
    return command


def build_guard_run_command(
    *,
    python_exe: str,
    run_dir: Path,
    required_anchors: Sequence[str] | None,
    min_latest_anchor_score: float,
    max_latest_drop: float,
    require_promotion_pass_after_attempts: int,
    max_consecutive_promotion_failures: int,
    max_vtrace_rho_p99: float | None,
) -> list[str]:
    command = [
        python_exe,
        "python/scripts/learning_progress_diagnostic.py",
        "--run-dir",
        run_dir.as_posix(),
        "--league-guard",
        "--guard-min-latest-anchor-score",
        str(float(min_latest_anchor_score)),
        "--guard-max-latest-drop",
        str(float(max_latest_drop)),
        "--guard-require-promotion-pass-after-attempts",
        str(int(require_promotion_pass_after_attempts)),
        "--guard-max-consecutive-promotion-failures",
        str(int(max_consecutive_promotion_failures)),
    ]
    resolved_required_anchors = tuple(required_anchors or DEFAULT_GUARD_REQUIRED_ANCHORS)
    for anchor in resolved_required_anchors:
        command.extend(["--guard-required-anchor", anchor])
    if max_vtrace_rho_p99 is not None:
        command.extend(["--guard-max-vtrace-rho-p99", str(float(max_vtrace_rho_p99))])
    return command


def build_guided_bootstrap_loop_command(
    *,
    python_exe: str,
    initial_run_dir: Path,
    initial_policy_id: str,
    seed_run_dir: Path | None,
    run_prefix: str,
    stack_config: Path,
    alias_policy_id: str,
    segments: int,
    segment_updates: int,
    confirm_paired_seeds: int,
    stop_on_latest_falloff: bool,
) -> list[str]:
    command = [
        python_exe,
        "python/scripts/segmented_b1_guided_bootstrap.py",
        "--initial-run-dir",
        initial_run_dir.as_posix(),
        "--initial-policy-id",
        str(initial_policy_id),
        "--run-prefix",
        str(run_prefix),
        "--stack-config",
        stack_config.as_posix(),
        "--alias-policy-id",
        str(alias_policy_id),
        "--segments",
        str(int(segments)),
        "--segment-updates",
        str(int(segment_updates)),
        "--confirm-paired-seeds",
        str(int(confirm_paired_seeds)),
    ]
    if seed_run_dir is not None:
        command.extend(["--seed-run-dir", seed_run_dir.as_posix()])
    if stop_on_latest_falloff:
        command.append("--stop-on-latest-falloff")
    return command


def build_guarded_league_bootstrap_command(
    *,
    python_exe: str,
    init_from_checkpoint: Path,
    seed_snapshot_run_dir: Path,
    run_prefix: str,
    stack_config: Path,
    segments: int,
    segment_updates: int,
    first_init_schedule_offset_updates: int | None,
    confirm_paired_seeds: int,
    publish_min_confirm_paired_seeds: int,
    confirm_recent_candidate_count: int,
    reference_summary_json: Path | None,
    multiobjective_reference_summary_jsons: tuple[Path, ...],
    multiobjective_fixed_opponents: tuple[str, ...],
    learned_guard_opponents: tuple[str, ...],
    min_learned_guard_mean: float,
    min_learned_guard_reference_delta: float,
    reference_label: str,
    min_required_anchor_score: float,
    max_reference_drop: float,
    selected_alias_policy_id: str,
) -> list[str]:
    command = [
        python_exe,
        "python/scripts/guarded_league_bootstrap.py",
        "--init-from-checkpoint",
        init_from_checkpoint.as_posix(),
        "--seed-snapshot-run-dir",
        seed_snapshot_run_dir.as_posix(),
        "--run-prefix",
        str(run_prefix),
        "--stack-config",
        stack_config.as_posix(),
        "--segments",
        str(int(segments)),
        "--segment-updates",
        str(int(segment_updates)),
        "--confirm-paired-seeds",
        str(int(confirm_paired_seeds)),
        "--publish-min-confirm-paired-seeds",
        str(int(publish_min_confirm_paired_seeds)),
        "--confirm-recent-candidate-count",
        str(int(confirm_recent_candidate_count)),
        "--min-required-anchor-score",
        str(float(min_required_anchor_score)),
        "--max-reference-drop",
        str(float(max_reference_drop)),
        "--selected-alias-policy-id",
        str(selected_alias_policy_id),
    ]
    if first_init_schedule_offset_updates is not None:
        command.extend(["--first-init-schedule-offset-updates", str(int(first_init_schedule_offset_updates))])
    if reference_summary_json is not None:
        command.extend(
            [
                "--reference-summary-json",
                reference_summary_json.as_posix(),
                "--reference-label",
                str(reference_label),
            ]
        )
    for path in multiobjective_reference_summary_jsons:
        command.extend(["--multiobjective-reference-summary-json", path.as_posix()])
    for opponent in multiobjective_fixed_opponents:
        command.extend(["--multiobjective-fixed-opponent", str(opponent)])
    for opponent in learned_guard_opponents:
        command.extend(["--learned-guard-opponent", str(opponent)])
    if learned_guard_opponents:
        command.extend(["--min-learned-guard-mean", str(float(min_learned_guard_mean))])
        command.extend(["--min-learned-guard-reference-delta", str(float(min_learned_guard_reference_delta))])
    return command
