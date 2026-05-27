from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import DEFAULT_REQUIRED_ANCHORS, load_reference_anchor_scores
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_NAME
from weiss_rl.experiments.main_league_multiobjective_gate import (
    FIXED_THESIS_OPPONENTS,
    MultiObjectiveGateConfig,
    evaluate_main_league_multiobjective_gate,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap import repo_relative, resolve_snapshot_checkpoint_path

DEFAULT_STACK_CONFIG = Path(
    "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
)
DEFAULT_RUN_PREFIX = "guarded_league_bootstrap"
DEFAULT_SELECTED_ALIAS_POLICY_ID = "main_league_selected"
DEFAULT_CONFIRM_OPPONENTS = (NOLEAGUE_BASELINE_NAME, *DEFAULT_REQUIRED_ANCHORS)
DEFAULT_RUNTIME_OVERRIDES = (
    "league.pool.seed_snapshot_champion_import=pinned",
    "league.pool.seed_snapshot_import_filter=pinned",
)


def runtime_overrides_with_defaults(
    overrides: Sequence[str] | None,
    *,
    apply_seed_snapshot_defaults: bool = True,
) -> tuple[str, ...]:
    requested = tuple(str(override) for override in (overrides or ()))
    requested_keys = {override.split("=", 1)[0].strip() for override in requested}
    defaults = (
        tuple(
            override
            for override in DEFAULT_RUNTIME_OVERRIDES
            if override.split("=", 1)[0].strip() not in requested_keys
        )
        if apply_seed_snapshot_defaults
        else ()
    )
    return (*defaults, *requested)


@dataclass(frozen=True, slots=True)
class LeagueSegmentRuntime:
    num_envs: int = 288
    unroll_length: int = 64
    segment_updates: int = 10
    runtime_mode: str = "train_async_fast"
    simulator_profile: str = "fast"
    device: str = "cuda"
    checkpoint_interval_updates: int = 5
    collection_backend: str = "process"
    profile_timers: bool = True
    overrides: tuple[str, ...] = DEFAULT_RUNTIME_OVERRIDES


@dataclass(frozen=True, slots=True)
class GuardedLeagueBootstrapConfig:
    repo_root: Path
    init_checkpoint_path: Path
    seed_snapshot_run_dir: Path
    b1_baseline_run_dir: Path | None = None
    run_prefix: str = DEFAULT_RUN_PREFIX
    stack_config: Path = DEFAULT_STACK_CONFIG
    segments: int = 4
    runtime: LeagueSegmentRuntime = LeagueSegmentRuntime()
    first_init_schedule_offset_updates: int | None = None
    confirm_paired_seeds: int = 64
    publish_min_confirm_paired_seeds: int = 256
    confirm_recent_candidate_count: int = 1
    bootstrap_samples: int = 2000
    required_anchors: tuple[str, ...] = DEFAULT_REQUIRED_ANCHORS
    confirm_opponents: tuple[str, ...] = DEFAULT_CONFIRM_OPPONENTS
    min_required_anchor_score: float = 0.5
    reference_anchor_scores: Mapping[str, float] = field(default_factory=dict)
    multiobjective_reference_summary_jsons: tuple[Path, ...] = ()
    multiobjective_fixed_opponents: tuple[str, ...] = FIXED_THESIS_OPPONENTS
    learned_guard_opponents: tuple[str, ...] = ()
    min_multiobjective_fixed_score: float = 0.5
    max_multiobjective_fixed_reference_drop: float = 0.0
    min_learned_guard_score: float = 0.5
    min_learned_guard_mean: float = 0.5
    min_learned_guard_reference_delta: float | None = 0.0
    max_learned_guard_reference_drop: float | None = None
    reference_label: str = "reference"
    max_reference_drop: float = 0.04
    selected_alias_policy_id: str = DEFAULT_SELECTED_ALIAS_POLICY_ID
    continue_unpublished_confirmed: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    policy_id: str
    update: int
    checkpoint_path: Path


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


def build_train_segment_command(
    *,
    config: GuardedLeagueBootstrapConfig,
    segment_run_label: str,
    init_checkpoint_path: Path,
    init_schedule_offset_updates: int | None = None,
) -> list[str]:
    runtime = config.runtime
    b1_baseline_run_dir = config.b1_baseline_run_dir or config.seed_snapshot_run_dir
    command = [
        sys.executable,
        "python/scripts/train.py",
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--run-label",
        segment_run_label,
        "--num-envs",
        str(int(runtime.num_envs)),
        "--unroll-length",
        str(int(runtime.unroll_length)),
        "--max-updates",
        str(int(runtime.segment_updates)),
        "--runtime-mode",
        runtime.runtime_mode,
        "--profile",
        runtime.simulator_profile,
        "--device",
        runtime.device,
        "--checkpoint-interval-updates",
        str(int(runtime.checkpoint_interval_updates)),
        "--seed-snapshot-run-dir",
        repo_relative(config.seed_snapshot_run_dir, repo_root=config.repo_root).as_posix(),
        "--b1-baseline-run-dir",
        repo_relative(b1_baseline_run_dir, repo_root=config.repo_root).as_posix(),
        "--init-from-checkpoint",
        repo_relative(init_checkpoint_path, repo_root=config.repo_root).as_posix(),
        "--override",
        f"system.collection_backend={runtime.collection_backend}",
    ]
    if init_schedule_offset_updates is not None:
        command.extend(["--init-schedule-offset-updates", str(int(init_schedule_offset_updates))])
    if runtime.profile_timers:
        command.extend(["--override", "training.profile_timers=true"])
    for override in runtime.overrides:
        command.extend(["--override", str(override)])
    return command


def build_selector_command(
    *,
    config: GuardedLeagueBootstrapConfig,
    run_dir: Path,
    output_json: Path,
    publish_alias: bool,
) -> list[str]:
    command = [
        sys.executable,
        "python/scripts/select_b1_candidate.py",
        "--run-dir",
        repo_relative(run_dir, repo_root=config.repo_root).as_posix(),
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--min-required-anchor-score",
        str(float(config.min_required_anchor_score)),
        "--confirm-paired-seeds",
        str(int(config.confirm_paired_seeds)),
        "--output-json",
        repo_relative(output_json, repo_root=config.repo_root).as_posix(),
    ]
    for anchor in config.required_anchors:
        command.extend(["--required-anchor", anchor])
    for opponent in config.confirm_opponents:
        command.extend(["--confirm-opponent", opponent])
    if publish_alias:
        command.extend(
            [
                "--publish-selected-alias",
                "--selected-alias-policy-id",
                str(config.selected_alias_policy_id),
            ]
        )
    return command


def build_targeted_confirm_command(
    *,
    config: GuardedLeagueBootstrapConfig,
    run_dir: Path,
    focal_policy_id: str,
    output_subdir: str,
) -> list[str]:
    b1_baseline_run_dir = config.b1_baseline_run_dir or run_dir
    command = [
        sys.executable,
        "python/scripts/targeted_confirm_eval.py",
        "--stack-config",
        repo_relative(config.stack_config, repo_root=config.repo_root).as_posix(),
        "--run-dir",
        repo_relative(run_dir, repo_root=config.repo_root).as_posix(),
        "--snapshot-registry-json",
        repo_relative(run_dir / "training" / "snapshots" / "registry.json", repo_root=config.repo_root).as_posix(),
        "--b1-baseline-run-dir",
        repo_relative(b1_baseline_run_dir, repo_root=config.repo_root).as_posix(),
        "--focal-policy-id",
        focal_policy_id,
        "--paired-seeds",
        str(int(config.confirm_paired_seeds)),
        "--workers",
        "1",
        "--bootstrap-samples",
        str(int(config.bootstrap_samples)),
        "--output-subdir",
        output_subdir,
    ]
    for opponent in _effective_confirm_opponents(config):
        command.extend(["--opponent", opponent])
    return command


def _effective_confirm_opponents(config: GuardedLeagueBootstrapConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*config.confirm_opponents, *_effective_learned_guard_opponents(config)]))


def _effective_learned_guard_opponents(config: GuardedLeagueBootstrapConfig) -> tuple[str, ...]:
    configured = tuple(dict.fromkeys(str(opponent) for opponent in config.learned_guard_opponents))
    if configured:
        return configured
    fixed_opponents = tuple(str(opponent) for opponent in config.multiobjective_fixed_opponents)
    return tuple(
        dict.fromkeys(
            opponent
            for opponent in config.confirm_opponents
            if not any(_is_seed_wrapped_suffix_match(str(opponent), fixed) for fixed in fixed_opponents)
        )
    )


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def policy_snapshots(run_dir: Path) -> list[SnapshotCandidate]:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    payload = _read_json_object(registry_path)
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError(f"snapshot registry contains no snapshots: {registry_path}")
    candidates: list[SnapshotCandidate] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        policy_id = str(snapshot.get("policy_id", "")).strip()
        update = snapshot.get("update", snapshot.get("update_count"))
        if not policy_id.startswith("policy_") or isinstance(update, bool) or not isinstance(update, int):
            continue
        checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{int(update)}.pt"
        if checkpoint_path.is_file():
            candidates.append(
                SnapshotCandidate(policy_id=policy_id, update=int(update), checkpoint_path=checkpoint_path)
            )
    if not candidates:
        raise ValueError(f"no train policy snapshots with checkpoints found in {registry_path}")
    return sorted(candidates, key=lambda item: (item.update, item.policy_id))


def latest_policy_snapshot(run_dir: Path) -> SnapshotCandidate:
    return policy_snapshots(run_dir)[-1]


def recent_policy_snapshots(run_dir: Path, *, count: int) -> list[SnapshotCandidate]:
    if int(count) < 1:
        raise ValueError("count must be >= 1")
    snapshots = policy_snapshots(run_dir)
    return snapshots[-int(count) :]


def targeted_confirm_summary_path(*, run_dir: Path, output_subdir: str, paired_seeds: int) -> Path:
    return run_dir / "eval" / output_subdir / f"targeted_confirm{int(paired_seeds)}_summary.json"


def load_targeted_confirm_scores(path: Path) -> dict[str, float]:
    payload = _read_json_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"targeted confirm summary missing rows: {path}")
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        opponent = row.get("opponent_policy_id")
        mean = row.get("mean")
        if isinstance(opponent, str) and opponent and isinstance(mean, int | float):
            scores[opponent] = float(mean)
    if not scores:
        raise ValueError(f"targeted confirm summary has no opponent scores: {path}")
    return scores


def selected_candidate_or_none(path: Path) -> dict[str, Any] | None:
    payload = _read_json_object(path)
    selected = payload.get("selected")
    return dict(selected) if isinstance(selected, Mapping) else None


def selected_candidate(path: Path) -> dict[str, Any]:
    selected = selected_candidate_or_none(path)
    if selected is None:
        raise RuntimeError(f"candidate selector did not produce a selected candidate: {path}")
    return selected


def selection_anchor_scores(candidate: Mapping[str, Any]) -> dict[str, float]:
    for key in ("selection_anchor_scores", "anchor_scores"):
        raw_scores = candidate.get(key)
        if not isinstance(raw_scores, Mapping):
            continue
        scores: dict[str, float] = {}
        for anchor, value in raw_scores.items():
            if isinstance(anchor, str) and anchor and isinstance(value, int | float) and not isinstance(value, bool):
                scores[anchor] = float(value)
        if scores:
            return scores
    return {}


def evaluate_guard(
    *,
    scores: Mapping[str, float],
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
    reference_anchor_scores: Mapping[str, float],
    max_reference_drop: float,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    required_payload: dict[str, dict[str, float | None]] = {}
    for anchor in required_anchors:
        score = scores.get(anchor)
        reference = reference_anchor_scores.get(anchor)
        delta = None if score is None or reference is None else float(score) - float(reference)
        required_payload[anchor] = {
            "score": None if score is None else float(score),
            "reference": None if reference is None else float(reference),
            "delta": delta,
        }
        if score is None:
            failures.append({"anchor": anchor, "reason": "missing_score"})
            continue
        if float(score) < float(min_required_anchor_score):
            failures.append(
                {
                    "anchor": anchor,
                    "reason": "below_min_required_anchor_score",
                    "score": float(score),
                    "threshold": float(min_required_anchor_score),
                }
            )
        if reference is not None and delta is not None and delta < -float(max_reference_drop):
            failures.append(
                {
                    "anchor": anchor,
                    "reason": "below_reference_drop_limit",
                    "score": float(score),
                    "reference": float(reference),
                    "delta": float(delta),
                    "threshold": -float(max_reference_drop),
                }
            )
    return {
        "passed": not failures,
        "failures": failures,
        "required_anchor_scores": required_payload,
        "min_required_anchor_score": float(min_required_anchor_score),
        "max_reference_drop": float(max_reference_drop),
    }


def evaluate_multiobjective_guard(
    *,
    candidate_summary_json: Path,
    reference_summary_jsons: Sequence[Path],
    fixed_opponents: Sequence[str],
    learned_opponents: Sequence[str],
    min_fixed_score: float,
    max_fixed_reference_drop: float,
    min_learned_score: float,
    min_learned_mean: float,
    min_learned_reference_delta: float | None,
    max_learned_reference_drop: float | None,
) -> dict[str, Any] | None:
    if not learned_opponents:
        return None
    return evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(Path(candidate_summary_json),),
            reference_summary_jsons=tuple(Path(path) for path in reference_summary_jsons),
            fixed_opponents=tuple(str(opponent) for opponent in fixed_opponents),
            learned_opponents=tuple(str(opponent) for opponent in learned_opponents),
            min_fixed_score=float(min_fixed_score),
            max_fixed_reference_drop=float(max_fixed_reference_drop),
            min_learned_score=float(min_learned_score),
            min_learned_mean=float(min_learned_mean),
            min_learned_reference_delta=min_learned_reference_delta,
            max_learned_reference_drop=max_learned_reference_drop,
        )
    )


def run_guarded_league_bootstrap(
    config: GuardedLeagueBootstrapConfig,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    repo_root = config.repo_root.resolve()
    diagnostics_dir = repo_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{config.run_prefix}_guarded_league_bootstrap_summary.json"
    current_checkpoint = config.init_checkpoint_path.resolve()
    effective_learned_guard_opponents = _effective_learned_guard_opponents(config)
    summary: dict[str, Any] = {
        "kind": "guarded_league_bootstrap_v1",
        "created_unix": time.time(),
        "repo_root": repo_root.as_posix(),
        "stack_config": repo_relative(config.stack_config, repo_root=repo_root).as_posix(),
        "seed_snapshot_run_dir": repo_relative(config.seed_snapshot_run_dir, repo_root=repo_root).as_posix(),
        "b1_baseline_run_dir": repo_relative(
            config.b1_baseline_run_dir or config.seed_snapshot_run_dir,
            repo_root=repo_root,
        ).as_posix(),
        "initial_checkpoint": repo_relative(config.init_checkpoint_path, repo_root=repo_root).as_posix(),
        "run_prefix": config.run_prefix,
        "segments_requested": int(config.segments),
        "confirm_paired_seeds": int(config.confirm_paired_seeds),
        "publish_min_confirm_paired_seeds": int(config.publish_min_confirm_paired_seeds),
        "confirm_recent_candidate_count": int(config.confirm_recent_candidate_count),
        "required_anchors": list(config.required_anchors),
        "confirm_opponents": list(config.confirm_opponents),
        "effective_confirm_opponents": list(_effective_confirm_opponents(config)),
        "min_required_anchor_score": float(config.min_required_anchor_score),
        "first_init_schedule_offset_updates": config.first_init_schedule_offset_updates,
        "reference_label": str(config.reference_label),
        "reference_anchor_scores": {key: float(value) for key, value in sorted(config.reference_anchor_scores.items())},
        "multiobjective_reference_summary_jsons": [
            repo_relative(path, repo_root=repo_root).as_posix()
            for path in config.multiobjective_reference_summary_jsons
        ],
        "multiobjective_fixed_opponents": list(config.multiobjective_fixed_opponents),
        "configured_learned_guard_opponents": list(config.learned_guard_opponents),
        "learned_guard_opponents": list(effective_learned_guard_opponents),
        "learned_guard_opponents_inferred": not bool(config.learned_guard_opponents)
        and bool(effective_learned_guard_opponents),
        "multiobjective_thresholds": {
            "min_fixed_score": float(config.min_multiobjective_fixed_score),
            "max_fixed_reference_drop": float(config.max_multiobjective_fixed_reference_drop),
            "min_learned_score": float(config.min_learned_guard_score),
            "min_learned_mean": float(config.min_learned_guard_mean),
            "min_learned_reference_delta": config.min_learned_guard_reference_delta,
            "max_learned_reference_drop": config.max_learned_guard_reference_drop,
        },
        "max_reference_drop": float(config.max_reference_drop),
        "selected_alias_policy_id": str(config.selected_alias_policy_id),
        "continue_unpublished_confirmed": bool(config.continue_unpublished_confirmed),
        "segments": [],
        "status": "planned" if config.dry_run else "running",
    }

    for segment_index in range(1, int(config.segments) + 1):
        segment_run_label = f"{config.run_prefix}_seg{segment_index:02d}"
        segment_run_dir = repo_root / "runs" / segment_run_label
        train_command = build_train_segment_command(
            config=config,
            segment_run_label=segment_run_label,
            init_checkpoint_path=current_checkpoint,
            init_schedule_offset_updates=(
                config.first_init_schedule_offset_updates if int(segment_index) == 1 else None
            ),
        )
        segment_record: dict[str, Any] = {
            "segment": int(segment_index),
            "run_label": segment_run_label,
            "run_dir": repo_relative(segment_run_dir, repo_root=repo_root).as_posix(),
            "source_checkpoint": repo_relative(current_checkpoint, repo_root=repo_root).as_posix(),
            "train_command": _command_record(train_command),
        }
        preselect_json = diagnostics_dir / f"{segment_run_label}_candidate_preconfirm.json"
        final_selection_json = diagnostics_dir / f"{segment_run_label}_candidate_selection.json"
        publish_selection_json = diagnostics_dir / f"{segment_run_label}_candidate_published.json"
        segment_record["preselect_json"] = repo_relative(preselect_json, repo_root=repo_root).as_posix()
        segment_record["final_selection_json"] = repo_relative(final_selection_json, repo_root=repo_root).as_posix()
        segment_record["publish_selection_json"] = repo_relative(publish_selection_json, repo_root=repo_root).as_posix()
        summary["segments"].append(segment_record)

        if config.dry_run:
            segment_record["status"] = "planned"
            segment_record["preselect_command"] = _command_record(
                build_selector_command(
                    config=config,
                    run_dir=segment_run_dir,
                    output_json=preselect_json,
                    publish_alias=False,
                )
            )
            segment_record["targeted_confirm_command_template"] = _command_record(
                build_targeted_confirm_command(
                    config=config,
                    run_dir=segment_run_dir,
                    focal_policy_id="<candidate-policy-id>",
                    output_subdir=(f"guard_confirm{int(config.confirm_paired_seeds)}_<candidate-policy-id>"),
                )
            )
            segment_record["confirm_recent_candidate_count"] = int(config.confirm_recent_candidate_count)
            segment_record["final_selector_command"] = _command_record(
                build_selector_command(
                    config=config,
                    run_dir=segment_run_dir,
                    output_json=final_selection_json,
                    publish_alias=False,
                )
            )
            segment_record["publish_selector_command"] = _command_record(
                build_selector_command(
                    config=config,
                    run_dir=segment_run_dir,
                    output_json=publish_selection_json,
                    publish_alias=True,
                )
            )
            break

        _run_command(train_command, cwd=repo_root, runner=runner, env=_fixed_hash_seed_env())
        candidate_limit = int(config.confirm_recent_candidate_count)
        recent_candidates = recent_policy_snapshots(segment_run_dir, count=candidate_limit)
        latest = latest_policy_snapshot(segment_run_dir)
        preselect_command = build_selector_command(
            config=config,
            run_dir=segment_run_dir,
            output_json=preselect_json,
            publish_alias=False,
        )
        segment_record["preselect_command"] = _command_record(preselect_command)
        _run_command(preselect_command, cwd=repo_root, runner=runner)
        preselected = selected_candidate_or_none(preselect_json)
        focal_policy_ids: list[str] = []
        preselected_policy_id = (
            str(preselected.get("snapshot_policy_id", "")).strip() if preselected is not None else ""
        )
        if preselected_policy_id:
            focal_policy_ids.append(preselected_policy_id)
        for candidate in recent_candidates:
            if len(focal_policy_ids) >= candidate_limit:
                break
            if candidate.policy_id not in focal_policy_ids:
                focal_policy_ids.append(candidate.policy_id)
        if not focal_policy_ids:
            focal_policy_ids.append(latest.policy_id)

        segment_record["latest_policy_id"] = latest.policy_id
        segment_record["latest_update"] = int(latest.update)
        segment_record["latest_checkpoint"] = repo_relative(latest.checkpoint_path, repo_root=repo_root).as_posix()
        segment_record["confirm_recent_candidate_count"] = int(config.confirm_recent_candidate_count)
        segment_record["confirm_focal_policy_ids"] = list(focal_policy_ids)
        if preselected is not None:
            segment_record["preselected"] = preselected
        confirm_env = os.environ.copy()
        confirm_env["PYTHONHASHSEED"] = "0"
        targeted_confirm_records: list[dict[str, Any]] = []
        for focal_policy_id in focal_policy_ids:
            confirm_subdir = f"guard_confirm{int(config.confirm_paired_seeds)}_{focal_policy_id}"
            confirm_command = build_targeted_confirm_command(
                config=config,
                run_dir=segment_run_dir,
                focal_policy_id=focal_policy_id,
                output_subdir=confirm_subdir,
            )
            confirm_record: dict[str, Any] = {
                "focal_policy_id": focal_policy_id,
                "output_subdir": confirm_subdir,
                "command": _command_record(confirm_command),
            }
            targeted_confirm_records.append(confirm_record)
            _run_command(confirm_command, cwd=repo_root, runner=runner, env=confirm_env)
            confirm_summary_path = targeted_confirm_summary_path(
                run_dir=segment_run_dir,
                output_subdir=confirm_subdir,
                paired_seeds=config.confirm_paired_seeds,
            )
            confirm_record["summary_path"] = repo_relative(confirm_summary_path, repo_root=repo_root).as_posix()
            confirm_record["anchor_scores"] = load_targeted_confirm_scores(confirm_summary_path)
        segment_record["targeted_confirm_commands"] = [
            dict(record["command"]) for record in targeted_confirm_records if isinstance(record.get("command"), Mapping)
        ]
        segment_record["targeted_confirm_records"] = targeted_confirm_records
        if targeted_confirm_records:
            first_confirm = targeted_confirm_records[0]
            segment_record["confirm_focal_policy_id"] = first_confirm["focal_policy_id"]
            segment_record["targeted_confirm_command"] = first_confirm["command"]
        final_selector_command = build_selector_command(
            config=config,
            run_dir=segment_run_dir,
            output_json=final_selection_json,
            publish_alias=False,
        )
        segment_record["final_selector_command"] = _command_record(final_selector_command)
        _run_command(final_selector_command, cwd=repo_root, runner=runner)
        final_selected = selected_candidate(final_selection_json)
        scores = selection_anchor_scores(final_selected)
        selected_confirm_summary_path = str(final_selected.get("selection_confirmation_summary_path") or "").strip()
        if not scores:
            for record in targeted_confirm_records:
                if str(record.get("summary_path", "")) == selected_confirm_summary_path:
                    scores = dict(record.get("anchor_scores", {}))
                    break
        if not scores and targeted_confirm_records:
            scores = dict(targeted_confirm_records[-1].get("anchor_scores", {}))
        guard = evaluate_guard(
            scores=scores,
            required_anchors=config.required_anchors,
            min_required_anchor_score=config.min_required_anchor_score,
            reference_anchor_scores=config.reference_anchor_scores,
            max_reference_drop=config.max_reference_drop,
        )
        if selected_confirm_summary_path:
            segment_record["targeted_confirm_summary"] = repo_relative(
                repo_root / selected_confirm_summary_path,
                repo_root=repo_root,
            ).as_posix()
        elif targeted_confirm_records:
            segment_record["targeted_confirm_summary"] = targeted_confirm_records[-1].get("summary_path")
        segment_record["targeted_anchor_scores"] = scores
        segment_record["selected"] = final_selected
        segment_record["anchor_scores"] = scores
        segment_record["guard"] = guard
        multiobjective_guard = None
        multiobjective_summary_path = _selected_confirm_summary_path(
            raw_path=selected_confirm_summary_path,
            fallback_record=targeted_confirm_records[-1] if targeted_confirm_records else None,
            repo_root=repo_root,
        )
        if multiobjective_summary_path is not None:
            multiobjective_guard = evaluate_multiobjective_guard(
                candidate_summary_json=multiobjective_summary_path,
                reference_summary_jsons=tuple(
                    _resolve_repo_path(path, repo_root=repo_root)
                    for path in config.multiobjective_reference_summary_jsons
                ),
                fixed_opponents=config.multiobjective_fixed_opponents,
                learned_opponents=effective_learned_guard_opponents,
                min_fixed_score=config.min_multiobjective_fixed_score,
                max_fixed_reference_drop=config.max_multiobjective_fixed_reference_drop,
                min_learned_score=config.min_learned_guard_score,
                min_learned_mean=config.min_learned_guard_mean,
                min_learned_reference_delta=config.min_learned_guard_reference_delta,
                max_learned_reference_drop=config.max_learned_guard_reference_drop,
            )
        if multiobjective_guard is not None:
            multiobjective_path = diagnostics_dir / f"{segment_run_label}_multiobjective_gate.json"
            multiobjective_path.write_text(
                json.dumps(multiobjective_guard, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            segment_record["multiobjective_guard"] = multiobjective_guard
            segment_record["multiobjective_guard_json"] = repo_relative(
                multiobjective_path,
                repo_root=repo_root,
            ).as_posix()
        if not bool(final_selected.get("eligible")):
            segment_record["status"] = "rejected"
            summary["status"] = "stopped_ineligible"
            summary["stop_reason"] = "selected checkpoint did not meet required anchor threshold"
            break
        if not bool(guard["passed"]):
            segment_record["status"] = "rejected"
            summary["status"] = "stopped_guard_failed"
            summary["stop_reason"] = "selected checkpoint failed B2/B3/B4 guard"
            break
        if multiobjective_guard is not None and not bool(multiobjective_guard["passed"]):
            segment_record["status"] = "rejected"
            summary["status"] = "stopped_multiobjective_guard_failed"
            summary["stop_reason"] = "selected checkpoint failed fixed/learned multi-objective guard"
            break
        if int(config.confirm_paired_seeds) < int(config.publish_min_confirm_paired_seeds):
            segment_record["status"] = "accepted_unpublished"
            segment_record["publish_skipped"] = {
                "reason": "confirmation_seed_count_below_publish_minimum",
                "confirm_paired_seeds": int(config.confirm_paired_seeds),
                "publish_min_confirm_paired_seeds": int(config.publish_min_confirm_paired_seeds),
                "continued_without_publish": bool(config.continue_unpublished_confirmed),
            }
            if bool(config.continue_unpublished_confirmed):
                selected_policy_id = str(final_selected.get("snapshot_policy_id", "")).strip()
                if not selected_policy_id:
                    raise RuntimeError(
                        f"candidate selector did not record a snapshot_policy_id: {final_selection_json}"
                    )
                selected_checkpoint = resolve_snapshot_checkpoint_path(
                    run_dir=segment_run_dir,
                    policy_id=selected_policy_id,
                )
                segment_record["selected_checkpoint"] = repo_relative(
                    selected_checkpoint, repo_root=repo_root
                ).as_posix()
                current_checkpoint = selected_checkpoint
                if int(segment_index) < int(config.segments):
                    continue
                summary["status"] = "completed_unpublished_confirmation_insufficient"
                summary["stop_reason"] = (
                    "all requested segments passed guard but were not published because confirmation seed count "
                    "is below publish_min_confirm_paired_seeds"
                )
                break
            summary["status"] = "stopped_publish_confirmation_insufficient"
            summary["stop_reason"] = (
                "selected checkpoint passed guard but was not published because confirmation seed count "
                "is below publish_min_confirm_paired_seeds"
            )
            break
        publish_selector_command = build_selector_command(
            config=config,
            run_dir=segment_run_dir,
            output_json=publish_selection_json,
            publish_alias=True,
        )
        segment_record["publish_selector_command"] = _command_record(publish_selector_command)
        _run_command(publish_selector_command, cwd=repo_root, runner=runner)
        published_selected = selected_candidate(publish_selection_json)
        selected_checkpoint = resolve_snapshot_checkpoint_path(
            run_dir=segment_run_dir,
            policy_id=str(config.selected_alias_policy_id),
        )
        segment_record["published_selected"] = published_selected
        segment_record["selected_alias_policy_id"] = str(config.selected_alias_policy_id)
        segment_record["selected_checkpoint"] = repo_relative(selected_checkpoint, repo_root=repo_root).as_posix()
        segment_record["status"] = "accepted"
        current_checkpoint = selected_checkpoint
        summary["status"] = "completed"

    summary["summary_path"] = summary_path.as_posix()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    return {"argv": list(command), "display": " ".join(f'"{part}"' if " " in part else part for part in command)}


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    runner: CommandRunner,
    env: Mapping[str, str] | None = None,
) -> None:
    completed = runner(list(command), cwd=cwd, check=False, env=None if env is None else dict(env))
    if int(completed.returncode) != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def _fixed_hash_seed_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    return env


def _selected_confirm_summary_path(
    *,
    raw_path: str,
    fallback_record: Mapping[str, Any] | None,
    repo_root: Path,
) -> Path | None:
    if raw_path:
        return _resolve_repo_path(Path(raw_path), repo_root=repo_root)
    if fallback_record is None:
        return None
    fallback = fallback_record.get("summary_path")
    if not isinstance(fallback, str) or not fallback.strip():
        return None
    return _resolve_repo_path(Path(fallback), repo_root=repo_root)


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_reference_scores_or_empty(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    return load_reference_anchor_scores(path)
