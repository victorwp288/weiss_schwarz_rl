from __future__ import annotations

from pathlib import Path


def _guarded_league_bootstrap_command(
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
        "-m",
        "weiss_rl.experiments.guarded_league_bootstrap_entrypoint",
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
