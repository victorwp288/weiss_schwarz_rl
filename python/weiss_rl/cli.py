from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
    SELECTED_CANDIDATE_POLICY_ID,
)


@dataclass(frozen=True, slots=True)
class TrainProfile:
    num_envs: int
    unroll_length: int
    max_updates: int
    runtime_mode: str
    simulator_profile: str
    device: str
    checkpoint_interval_updates: int | None
    overrides: tuple[str, ...] = ()


TRAIN_PROFILES: dict[str, TrainProfile] = {
    "smoke": TrainProfile(
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cpu",
        checkpoint_interval_updates=1,
        overrides=("system.collection_backend=auto",),
    ),
    "gpu-probe": TrainProfile(
        num_envs=32,
        unroll_length=16,
        max_updates=2,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=1,
        overrides=("system.collection_backend=auto", "training.profile_timers=true"),
    ),
    "league-probe": TrainProfile(
        num_envs=288,
        unroll_length=64,
        max_updates=50,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=5,
        overrides=("system.collection_backend=process", "training.profile_timers=true"),
    ),
    "thesis-local": TrainProfile(
        num_envs=288,
        unroll_length=64,
        max_updates=200,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=25,
        overrides=("system.collection_backend=auto",),
    ),
    "thesis-server": TrainProfile(
        num_envs=4096,
        unroll_length=64,
        max_updates=200,
        runtime_mode="train_async_fast",
        simulator_profile="fast",
        device="cuda",
        checkpoint_interval_updates=25,
        overrides=("system.collection_backend=process",),
    ),
}

B1_STACK_CONFIG = Path("configs/thesis/b1_noleague.yaml")
B1_GUIDED_SEED_STACK_CONFIG = Path("configs/thesis/b1_guided_seed.yaml")
MAIN_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
)
MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap.yaml")
MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_vtrace.yaml")
MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG = Path("configs/thesis/main_league_guided_bootstrap_seedchampion.yaml")
MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
)
MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml"
)
EVAL_STACK_CONFIG = Path("configs/thesis/final_eval.yaml")


def _repo_root(args_repo_root: Path | None) -> Path:
    return Path(__file__).resolve().parents[2] if args_repo_root is None else args_repo_root.resolve()


def _display(command: list[str]) -> str:
    return " ".join(command)


def _write_plan(*, repo_root: Path, name: str, command: list[str], payload: dict[str, Any]) -> None:
    plan_dir = repo_root / "runs" / "_workflow_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_payload = dict(payload)
    plan_payload["command"] = command
    plan_payload["cwd"] = repo_root.as_posix()
    plan_payload["status"] = "planned"
    (plan_dir / f"{name}.json").write_text(json.dumps(plan_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_or_plan(
    *,
    repo_root: Path,
    plan_name: str,
    command: list[str],
    dry_run: bool,
    payload: dict[str, Any],
) -> None:
    print(_display(command))
    if dry_run:
        _write_plan(repo_root=repo_root, name=plan_name, command=command, payload=payload)
        return
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _train_command(
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


def _run_relative_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _resolve_snapshot_checkpoint_path(*, repo_root: Path, run_dir: Path, policy_id: str) -> Path:
    resolved_run_dir = _run_relative_path(repo_root, run_dir)
    registry_path = resolved_run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise SystemExit(f"--init-from-run-dir snapshot registry not found: {registry_path}")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    if not isinstance(snapshots, list):
        raise SystemExit(f"snapshot registry must contain a snapshots list: {registry_path}")
    normalized_policy_id = str(policy_id).strip()
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if str(snapshot.get("policy_id", "")).strip() != normalized_policy_id:
            continue
        update = snapshot.get("update", snapshot.get("update_count"))
        if not isinstance(update, int):
            raise SystemExit(f"snapshot {normalized_policy_id!r} is missing an integer update in {registry_path}")
        checkpoint_path = resolved_run_dir / "training" / "checkpoints" / f"checkpoint_{int(update)}.pt"
        if not checkpoint_path.is_file():
            raise SystemExit(
                f"checkpoint for snapshot {normalized_policy_id!r} was not found: {checkpoint_path}. "
                "Use --init-from-checkpoint if the source checkpoint was moved."
            )
        return checkpoint_path
    raise SystemExit(f"snapshot policy id not found in {registry_path}: {normalized_policy_id}")


def _resolve_b1_seed_checkpoint_path(
    *,
    repo_root: Path,
    run_dir: Path,
    init_policy_id: str,
) -> tuple[Path, str]:
    requested_policy_id = str(init_policy_id).strip()
    policy_ids = (
        (SELECTED_CANDIDATE_POLICY_ID, NOLEAGUE_BASELINE_POLICY_ID, NOLEAGUE_BASELINE_NAME)
        if requested_policy_id in {"", "auto"}
        else (requested_policy_id,)
    )
    failures: list[str] = []
    for policy_id in policy_ids:
        try:
            return (
                _resolve_snapshot_checkpoint_path(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    policy_id=policy_id,
                ),
                policy_id,
            )
        except SystemExit as exc:
            failures.append(str(exc))
    raise SystemExit(
        "Could not resolve a B1 seed checkpoint from --b1-run. "
        "Tried policy ids: "
        f"{', '.join(policy_ids)}. "
        f"Last error: {failures[-1] if failures else 'none'}"
    )


def _guided_bootstrap_stack_config(
    *,
    vtrace_clamp: bool,
    seed_champions: bool,
    selected_seed_champion: bool,
) -> Path:
    selected_count = sum(bool(value) for value in (vtrace_clamp, seed_champions, selected_seed_champion))
    if selected_count > 1:
        raise SystemExit("--vtrace-clamp, --seed-champions, and --selected-seed-champion select distinct stacks")
    if selected_seed_champion:
        return MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG
    if seed_champions:
        return MAIN_GUIDED_BOOTSTRAP_SEEDCHAMPION_STACK_CONFIG
    if vtrace_clamp:
        return MAIN_GUIDED_BOOTSTRAP_VTRACE_STACK_CONFIG
    return MAIN_GUIDED_BOOTSTRAP_STACK_CONFIG


def _eval_command(
    *,
    python_exe: str,
    run_dir: Path,
    b1_baseline_run_dir: Path | None,
    smoke: bool,
) -> list[str]:
    command = [
        python_exe,
        "python/scripts/eval.py",
        "--stack-config",
        EVAL_STACK_CONFIG.as_posix(),
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


def _b2_audit_command(
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


def _guard_run_command(
    *,
    python_exe: str,
    run_dir: Path,
    required_anchors: tuple[str, ...],
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
    for anchor in required_anchors:
        command.extend(["--guard-required-anchor", anchor])
    if max_vtrace_rho_p99 is not None:
        command.extend(["--guard-max-vtrace-rho-p99", str(float(max_vtrace_rho_p99))])
    return command


def _guided_bootstrap_loop_command(
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


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print and save the command without executing it")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small thesis workflow command surface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_b1 = subparsers.add_parser("train-b1", help="Train the B1 NoLeague baseline")
    _add_common(train_b1)
    train_b1.add_argument("--run-label", required=True)
    train_b1.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_b1_guided = subparsers.add_parser(
        "train-b1-guided-seed",
        help="Train the guided B1-derived seed policy used for league bootstrap ablations",
    )
    _add_common(train_b1_guided)
    train_b1_guided.add_argument("--run-label", required=True)
    train_b1_guided.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_main = subparsers.add_parser("train-main", help="Train the main league thesis model")
    _add_common(train_main)
    train_main.add_argument("--run-label", required=True)
    train_main.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)
    train_main.add_argument("--seed-run", "--seed-snapshot-run-dir", dest="seed_snapshot_run_dir", type=Path)
    train_main.add_argument(
        "--init-policy-id",
        default="auto",
        help=(
            "B1 snapshot policy id used to initialize the main learner. "
            "Default auto tries selected_candidate, then canonical B1 aliases."
        ),
    )
    train_main.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    train_main_guided = subparsers.add_parser(
        "train-main-guided-bootstrap",
        help="Train the guarded guided-bootstrap league path from a confirmed seed checkpoint",
    )
    _add_common(train_main_guided)
    train_main_guided.add_argument("--run-label", required=True)
    train_main_guided.add_argument("--init-from-checkpoint", type=Path, default=None)
    train_main_guided.add_argument(
        "--init-from-run-dir",
        type=Path,
        default=None,
        help="Run directory whose snapshot registry should be used to resolve --init-policy-id.",
    )
    train_main_guided.add_argument(
        "--init-policy-id",
        type=str,
        default="",
        help="Snapshot policy id to initialize from, resolved to training/checkpoints/checkpoint_<update>.pt.",
    )
    train_main_guided.add_argument(
        "--seed-run",
        "--seed-snapshot-run-dir",
        dest="seed_snapshot_run_dir",
        type=Path,
        required=True,
    )
    train_main_guided.add_argument(
        "--b1-run",
        "--b1-baseline-run-dir",
        dest="b1_baseline_run_dir",
        type=Path,
        default=None,
        help="Optional strict B1 anchor; omitted for the current guided-bootstrap path.",
    )
    train_main_guided.add_argument(
        "--vtrace-clamp",
        action="store_true",
        help="Use the conservative V-trace-clipped guided-bootstrap stack.",
    )
    train_main_guided.add_argument(
        "--seed-champions",
        action="store_true",
        help=(
            "Treat imported seed snapshots as training-pool champions. This does not mark the run as thesis-promoted."
        ),
    )
    train_main_guided.add_argument(
        "--selected-seed-champion",
        action="store_true",
        help=(
            "Use the selected guided-bootstrap stack, where only pinned snapshots in --seed-run "
            "are imported as training-pool champions."
        ),
    )
    train_main_guided.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default="smoke")

    smoke_eval = subparsers.add_parser("smoke-eval", help="Run a tiny deterministic eval on a run directory")
    _add_common(smoke_eval)
    smoke_eval.add_argument("--run-dir", type=Path, required=True)
    smoke_eval.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, default=None)

    thesis_eval = subparsers.add_parser("eval-final", help="Run the thesis-grade final evaluation")
    _add_common(thesis_eval)
    thesis_eval.add_argument("--run-dir", type=Path, required=True)
    thesis_eval.add_argument("--b1-run", "--b1-baseline-run-dir", dest="b1_baseline_run_dir", type=Path, required=True)

    figures = subparsers.add_parser("figures", help="Export paper figures and tables for a run")
    _add_common(figures)
    figures.add_argument("--run-dir", type=Path, required=True)
    figures.add_argument("--fig-id", type=str, default="")
    figures.add_argument("--format", dest="formats", action="append", default=None)

    b2_audit = subparsers.add_parser("b2-audit", help="Run the standard learner-vs-B2 disagreement audit")
    _add_common(b2_audit)
    b2_audit.add_argument("--run-dir", type=Path, required=True)
    b2_audit.add_argument("--episodes-jsonl", type=Path, required=True)
    b2_audit.add_argument("--policy-id", required=True)
    b2_audit.add_argument("--output-run-dir", type=Path, default=None)
    b2_audit.add_argument("--snapshot-registry-json", type=Path, default=None)
    b2_audit.add_argument("--summary-json", type=Path, default=None)
    b2_audit.add_argument("--top-k", type=int, default=25)
    b2_audit.add_argument("--top-actions", type=int, default=5)
    b2_audit.add_argument("--allow-policy-id-mismatch", action="store_true")
    b2_audit.add_argument("--accept-snapshot-config-hash", action="append", default=[])

    guard_run = subparsers.add_parser("guard-run", help="Fail fast on unhealthy B1/main league probe artifacts")
    _add_common(guard_run)
    guard_run.add_argument("--run-dir", type=Path, required=True)
    guard_run.add_argument(
        "--required-anchor",
        action="append",
        default=None,
        help="Anchor that must remain above --min-latest-anchor-score; defaults to B2/B3/B4.",
    )
    guard_run.add_argument("--min-latest-anchor-score", type=float, default=0.45)
    guard_run.add_argument("--max-latest-drop", type=float, default=0.05)
    guard_run.add_argument("--require-promotion-pass-after-attempts", type=int, default=3)
    guard_run.add_argument("--max-consecutive-promotion-failures", type=int, default=3)
    guard_run.add_argument("--max-vtrace-rho-p99", type=float, default=None)

    guided_loop = subparsers.add_parser(
        "guided-bootstrap-loop",
        help="Run segmented guided-bootstrap continuation with automatic confirm/select/reanchor decisions",
    )
    _add_common(guided_loop)
    guided_loop.add_argument("--initial-run-dir", type=Path, required=True)
    guided_loop.add_argument("--initial-policy-id", default="guided_bootstrap_floor_selected")
    guided_loop.add_argument("--seed-run-dir", type=Path, default=None)
    guided_loop.add_argument("--run-prefix", default="b1_guided_floor_segmented")
    guided_loop.add_argument(
        "--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_ANCHOR_FLOOR_STACK_CONFIG
    )
    guided_loop.add_argument("--alias-policy-id", default="guided_bootstrap_floor_segmented_selected")
    guided_loop.add_argument("--segments", type=int, default=4)
    guided_loop.add_argument("--segment-updates", type=int, default=25)
    guided_loop.add_argument("--confirm-paired-seeds", type=int, default=64)
    guided_loop.add_argument("--stop-on-latest-falloff", action="store_true")

    guarded_league = subparsers.add_parser(
        "guarded-league-bootstrap",
        help="Run short guided-league segments, confirming B2/B3/B4 before advancing the selected checkpoint",
    )
    _add_common(guarded_league)
    guarded_league.add_argument("--init-from-checkpoint", type=Path, required=True)
    guarded_league.add_argument("--seed-snapshot-run-dir", type=Path, required=True)
    guarded_league.add_argument("--run-prefix", default="guarded_league_bootstrap")
    guarded_league.add_argument("--stack-config", type=Path, default=MAIN_GUIDED_BOOTSTRAP_SELECTED_STACK_CONFIG)
    guarded_league.add_argument("--segments", type=int, default=4)
    guarded_league.add_argument("--segment-updates", type=int, default=10)
    guarded_league.add_argument("--first-init-schedule-offset-updates", type=int, default=None)
    guarded_league.add_argument("--confirm-paired-seeds", type=int, default=64)
    guarded_league.add_argument("--publish-min-confirm-paired-seeds", type=int, default=256)
    guarded_league.add_argument(
        "--confirm-recent-candidate-count",
        type=int,
        default=1,
        help="Confirm this many recent train snapshots per segment before selecting the best confirmed checkpoint.",
    )
    guarded_league.add_argument("--reference-summary-json", type=Path, default=None)
    guarded_league.add_argument("--multiobjective-reference-summary-json", action="append", type=Path, default=[])
    guarded_league.add_argument("--multiobjective-fixed-opponent", action="append", default=[])
    guarded_league.add_argument("--learned-guard-opponent", action="append", default=[])
    guarded_league.add_argument("--min-learned-guard-mean", type=float, default=0.5)
    guarded_league.add_argument("--min-learned-guard-reference-delta", type=float, default=0.0)
    guarded_league.add_argument("--reference-label", default="reference")
    guarded_league.add_argument("--min-required-anchor-score", type=float, default=0.5)
    guarded_league.add_argument("--max-reference-drop", type=float, default=0.04)
    guarded_league.add_argument("--selected-alias-policy-id", default="main_league_selected")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root(args.repo_root)
    python_exe = sys.executable

    if args.command == "train-b1":
        profile = TRAIN_PROFILES[str(args.profile)]
        command = _train_command(
            python_exe=python_exe,
            stack_config=B1_STACK_CONFIG,
            run_label=str(args.run_label),
            profile=profile,
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=str(args.run_label),
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": "train-b1", "profile": str(args.profile)},
        )
        return

    if args.command == "train-b1-guided-seed":
        profile = TRAIN_PROFILES[str(args.profile)]
        command = _train_command(
            python_exe=python_exe,
            stack_config=B1_GUIDED_SEED_STACK_CONFIG,
            run_label=str(args.run_label),
            profile=profile,
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=str(args.run_label),
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": "train-b1-guided-seed", "profile": str(args.profile)},
        )
        return

    if args.command == "train-main":
        profile = TRAIN_PROFILES[str(args.profile)]
        init_from_checkpoint, resolved_init_policy_id = _resolve_b1_seed_checkpoint_path(
            repo_root=repo_root,
            run_dir=Path(args.b1_baseline_run_dir),
            init_policy_id=str(args.init_policy_id),
        )
        command = _train_command(
            python_exe=python_exe,
            stack_config=MAIN_STACK_CONFIG,
            run_label=str(args.run_label),
            profile=profile,
            b1_baseline_run_dir=Path(args.b1_baseline_run_dir),
            seed_snapshot_run_dir=args.seed_snapshot_run_dir,
            init_from_checkpoint=init_from_checkpoint,
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=str(args.run_label),
            command=command,
            dry_run=bool(args.dry_run),
            payload={
                "workflow": "train-main",
                "profile": str(args.profile),
                "init_policy_id": resolved_init_policy_id,
            },
        )
        return

    if args.command == "train-main-guided-bootstrap":
        profile = TRAIN_PROFILES[str(args.profile)]
        init_from_checkpoint = args.init_from_checkpoint
        if init_from_checkpoint is None:
            if args.init_from_run_dir is None or not str(args.init_policy_id).strip():
                raise SystemExit(
                    "train-main-guided-bootstrap requires either --init-from-checkpoint or "
                    "--init-from-run-dir plus --init-policy-id"
                )
            init_from_checkpoint = _resolve_snapshot_checkpoint_path(
                repo_root=repo_root,
                run_dir=Path(args.init_from_run_dir),
                policy_id=str(args.init_policy_id),
            )
        elif args.init_from_run_dir is not None or str(args.init_policy_id).strip():
            raise SystemExit("--init-from-checkpoint cannot be combined with --init-from-run-dir/--init-policy-id")
        command = _train_command(
            python_exe=python_exe,
            stack_config=_guided_bootstrap_stack_config(
                vtrace_clamp=bool(args.vtrace_clamp),
                seed_champions=bool(args.seed_champions),
                selected_seed_champion=bool(args.selected_seed_champion),
            ),
            run_label=str(args.run_label),
            profile=profile,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
            init_from_checkpoint=Path(init_from_checkpoint),
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=str(args.run_label),
            command=command,
            dry_run=bool(args.dry_run),
            payload={
                "workflow": "train-main-guided-bootstrap",
                "profile": str(args.profile),
                "vtrace_clamp": bool(args.vtrace_clamp),
                "seed_champions": bool(args.seed_champions),
                "selected_seed_champion": bool(args.selected_seed_champion),
                "init_policy_id": str(args.init_policy_id).strip() or None,
            },
        )
        return

    if args.command in {"smoke-eval", "eval-final"}:
        run_dir = Path(args.run_dir)
        command = _eval_command(
            python_exe=python_exe,
            run_dir=run_dir,
            b1_baseline_run_dir=args.b1_baseline_run_dir,
            smoke=args.command == "smoke-eval",
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{run_dir.name}_{args.command}",
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": str(args.command)},
        )
        return

    if args.command == "figures":
        run_dir = Path(args.run_dir)
        command = [python_exe, "python/scripts/make_figures.py", "--run-dir", run_dir.as_posix()]
        if str(args.fig_id).strip():
            command.extend(["--fig-id", str(args.fig_id).strip()])
        for fmt in args.formats or []:
            command.extend(["--format", str(fmt)])
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{run_dir.name}_figures",
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": "figures"},
        )
        return

    if args.command == "b2-audit":
        run_dir = Path(args.run_dir)
        command = _b2_audit_command(
            python_exe=python_exe,
            run_dir=run_dir,
            episodes_jsonl=Path(args.episodes_jsonl),
            policy_id=str(args.policy_id),
            output_run_dir=args.output_run_dir,
            snapshot_registry_json=args.snapshot_registry_json,
            summary_json=args.summary_json,
            top_k=int(args.top_k),
            top_actions=int(args.top_actions),
            allow_policy_id_mismatch=bool(args.allow_policy_id_mismatch),
            accepted_snapshot_config_hashes=tuple(str(value) for value in args.accept_snapshot_config_hash),
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{run_dir.name}_b2-audit",
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": "b2-audit"},
        )
        return

    if args.command == "guard-run":
        run_dir = Path(args.run_dir)
        command = _guard_run_command(
            python_exe=python_exe,
            run_dir=run_dir,
            required_anchors=tuple(
                args.required_anchor
                or (
                    "B2 HeuristicPublic",
                    "B3 HeuristicPublicAggro",
                    "B4 HeuristicPublicControl",
                )
            ),
            min_latest_anchor_score=float(args.min_latest_anchor_score),
            max_latest_drop=float(args.max_latest_drop),
            require_promotion_pass_after_attempts=int(args.require_promotion_pass_after_attempts),
            max_consecutive_promotion_failures=int(args.max_consecutive_promotion_failures),
            max_vtrace_rho_p99=args.max_vtrace_rho_p99,
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{run_dir.name}_guard-run",
            command=command,
            dry_run=bool(args.dry_run),
            payload={"workflow": "guard-run"},
        )
        return

    if args.command == "guided-bootstrap-loop":
        command = _guided_bootstrap_loop_command(
            python_exe=python_exe,
            initial_run_dir=Path(args.initial_run_dir),
            initial_policy_id=str(args.initial_policy_id),
            seed_run_dir=args.seed_run_dir,
            run_prefix=str(args.run_prefix),
            stack_config=Path(args.stack_config),
            alias_policy_id=str(args.alias_policy_id),
            segments=int(args.segments),
            segment_updates=int(args.segment_updates),
            confirm_paired_seeds=int(args.confirm_paired_seeds),
            stop_on_latest_falloff=bool(args.stop_on_latest_falloff),
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{args.run_prefix}_guided-bootstrap-loop",
            command=command,
            dry_run=bool(args.dry_run),
            payload={
                "workflow": "guided-bootstrap-loop",
                "initial_policy_id": str(args.initial_policy_id),
                "segments": int(args.segments),
                "segment_updates": int(args.segment_updates),
                "confirm_paired_seeds": int(args.confirm_paired_seeds),
            },
        )
        return

    if args.command == "guarded-league-bootstrap":
        if args.first_init_schedule_offset_updates is not None and int(args.first_init_schedule_offset_updates) < 0:
            raise SystemExit("--first-init-schedule-offset-updates must be >= 0")
        if int(args.publish_min_confirm_paired_seeds) < 1:
            raise SystemExit("--publish-min-confirm-paired-seeds must be >= 1")
        if int(args.confirm_recent_candidate_count) < 1:
            raise SystemExit("--confirm-recent-candidate-count must be >= 1")
        command = _guarded_league_bootstrap_command(
            python_exe=python_exe,
            init_from_checkpoint=Path(args.init_from_checkpoint),
            seed_snapshot_run_dir=Path(args.seed_snapshot_run_dir),
            run_prefix=str(args.run_prefix),
            stack_config=Path(args.stack_config),
            segments=int(args.segments),
            segment_updates=int(args.segment_updates),
            first_init_schedule_offset_updates=args.first_init_schedule_offset_updates,
            confirm_paired_seeds=int(args.confirm_paired_seeds),
            publish_min_confirm_paired_seeds=int(args.publish_min_confirm_paired_seeds),
            confirm_recent_candidate_count=int(args.confirm_recent_candidate_count),
            reference_summary_json=args.reference_summary_json,
            multiobjective_reference_summary_jsons=tuple(args.multiobjective_reference_summary_json or ()),
            multiobjective_fixed_opponents=tuple(args.multiobjective_fixed_opponent or ()),
            learned_guard_opponents=tuple(args.learned_guard_opponent or ()),
            min_learned_guard_mean=float(args.min_learned_guard_mean),
            min_learned_guard_reference_delta=float(args.min_learned_guard_reference_delta),
            reference_label=str(args.reference_label),
            min_required_anchor_score=float(args.min_required_anchor_score),
            max_reference_drop=float(args.max_reference_drop),
            selected_alias_policy_id=str(args.selected_alias_policy_id),
        )
        _run_or_plan(
            repo_root=repo_root,
            plan_name=f"{args.run_prefix}_guarded-league-bootstrap",
            command=command,
            dry_run=bool(args.dry_run),
            payload={
                "workflow": "guarded-league-bootstrap",
                "segments": int(args.segments),
                "segment_updates": int(args.segment_updates),
                "first_init_schedule_offset_updates": args.first_init_schedule_offset_updates,
                "confirm_paired_seeds": int(args.confirm_paired_seeds),
                "publish_min_confirm_paired_seeds": int(args.publish_min_confirm_paired_seeds),
                "confirm_recent_candidate_count": int(args.confirm_recent_candidate_count),
                "reference_label": str(args.reference_label) if args.reference_summary_json is not None else None,
                "multiobjective_reference_summary_jsons": [
                    path.as_posix() for path in tuple(args.multiobjective_reference_summary_json or ())
                ],
                "learned_guard_opponents": list(args.learned_guard_opponent or ()),
                "selected_alias_policy_id": str(args.selected_alias_policy_id),
            },
        )
        return

    raise AssertionError(f"Unhandled workflow command: {args.command}")


if __name__ == "__main__":
    main()
