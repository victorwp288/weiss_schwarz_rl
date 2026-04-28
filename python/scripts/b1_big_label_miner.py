"""Clustered B1 S1 counterfactual label miner.

This is a thin orchestration layer over ``b1_counterfactual_labels.py``.  The
single miner is already the source of truth for replay, forced actions, margin
labels, tensor records, and two-step search.  This script runs that miner across
predeclared target clusters so the label set grows beyond one brittle
pass-overextension pattern.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_STACK_CONFIG = "configs/residual_eval_s1.yaml"
DEFAULT_RUN_DIR = "runs/b1_s1_distillonly_u450_to_u455_20260427"
DEFAULT_B1_BASELINE_RUN_DIR = "runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425"
DEFAULT_B1_CHECKPOINT = (
    "runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/"
    "training/checkpoints/checkpoint_450.pt"
)
DEFAULT_LABEL_DIRS = [
    "runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_candidate_reps_p8_t24_a6_tensor_20260427",
    "runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_mainmove_no_pass_p8_t20_a2_stop1_20260427",
    "runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_pass_overextend_exclude2_p8_t80_a8_stop5_20260428",
    "runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_nonpass_exclude6_p16_t120_a8_stop3_20260428",
    "runs/b1_s1_distillonly_u450_to_u455_20260427/eval/b1_cf_labels_s1_pass_overextend_exclude7_p16_t100_a4_stop3_20260428",
]


@dataclass(frozen=True, slots=True)
class ClusterSpec:
    name: str
    target_families: str
    require_pass_legal: bool = False
    require_baseline_family: str = ""
    exclude_candidate_family: str = ""
    exclude_candidate_action_id: str = ""
    family_representatives_per_family: int = 1
    two_step_beam_targets: int = 0
    two_step_second_actions: int = 4
    two_step_min_first_delta: float = 0.05
    two_step_include_positive_first: bool = False


CLUSTERS: dict[str, ClusterSpec] = {
    "pass_overextend": ClusterSpec(
        name="pass_overextend",
        target_families="main_play_character",
        require_pass_legal=True,
        require_baseline_family="main_play_character",
    ),
    "main_nonpass": ClusterSpec(
        name="main_nonpass",
        target_families="main_play_character,main_move",
        exclude_candidate_family="pass",
        exclude_candidate_action_id="51",
        family_representatives_per_family=2,
    ),
    "attack_climax": ClusterSpec(
        name="attack_climax",
        target_families="attack,climax_play,main_play_event",
        family_representatives_per_family=2,
    ),
    "level_clock": ClusterSpec(
        name="level_clock",
        target_families="level_up,clock_from_hand",
        family_representatives_per_family=2,
    ),
    "broad_twostep": ClusterSpec(
        name="broad_twostep",
        target_families="main_play_character,main_move,attack,climax_play,main_play_event,level_up,clock_from_hand",
        family_representatives_per_family=2,
        two_step_beam_targets=8,
        two_step_second_actions=4,
        two_step_min_first_delta=0.03,
        two_step_include_positive_first=True,
    ),
}


def _script_path(name: str) -> str:
    return str(Path("python/scripts") / name)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_repeated(cmd: list[str], flag: str, values: Iterable[str]) -> None:
    for value in values:
        cmd.extend([flag, value])


def _default_tag() -> str:
    return datetime.now().strftime("biglabels_%Y%m%d_%H%M%S")


def _parse_clusters(value: str) -> list[ClusterSpec]:
    names = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not names:
        raise SystemExit("--clusters must name at least one cluster")
    missing = [name for name in names if name not in CLUSTERS]
    if missing:
        raise SystemExit(f"unknown label clusters: {', '.join(missing)}; known={', '.join(sorted(CLUSTERS))}")
    return [CLUSTERS[name] for name in names]


def _summarize_labels(label_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_dir: dict[str, int] = {}
    for label_dir in label_dirs:
        label_rows = _read_jsonl(label_dir / "counterfactual_labels.jsonl")
        by_dir[str(label_dir)] = len(label_rows)
        rows.extend(label_rows)
    family_counts: Counter[str] = Counter()
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    winner_flips = 0
    margin_positive = 0
    score_deltas: list[float] = []
    for row in rows:
        candidate = row.get("candidate_action")
        family = str(candidate.get("family") or row.get("positive_family") or "") if isinstance(candidate, Mapping) else ""
        family_counts[family or "unknown"] += 1
        if row.get("episode_seed") is not None:
            seeds.add(int(row["episode_seed"]))
        fingerprint = row.get("legal_action_fingerprint") or row.get("legal_ids_sha256")
        if fingerprint:
            fingerprints.add(str(fingerprint))
        if bool(row.get("winner_flipped")):
            winner_flips += 1
        if bool(row.get("margin_positive")):
            margin_positive += 1
        if row.get("score_delta") is not None:
            score_deltas.append(float(row["score_delta"]))
    return {
        "label_dirs": [str(path) for path in label_dirs],
        "labels_by_dir": by_dir,
        "label_count": len(rows),
        "winner_flip_labels": winner_flips,
        "margin_positive_labels": margin_positive,
        "unique_episode_seeds": len(seeds),
        "unique_legal_fingerprints": len(fingerprints),
        "positive_families": dict(sorted(family_counts.items())),
        "score_delta": {
            "min": min(score_deltas) if score_deltas else None,
            "max": max(score_deltas) if score_deltas else None,
            "mean": (sum(score_deltas) / len(score_deltas)) if score_deltas else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=_default_tag())
    parser.add_argument("--stack-config", default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--b1-baseline-run-dir", default=DEFAULT_B1_BASELINE_RUN_DIR)
    parser.add_argument("--checkpoint-policy", default=f"u450={DEFAULT_B1_CHECKPOINT}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label-dir", action="append", default=[])
    parser.add_argument(
        "--clusters",
        default="pass_overextend,main_nonpass,attack_climax,level_clock",
        help=f"Comma-separated clusters. Known: {', '.join(sorted(CLUSTERS))}",
    )
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--target-states", type=int, default=120)
    parser.add_argument("--targets-per-pair", type=int, default=4)
    parser.add_argument("--actions-per-state", type=int, default=8)
    parser.add_argument("--max-forced-replays", type=int, default=640)
    parser.add_argument("--stop-after-positives-per-cluster", type=int, default=5)
    parser.add_argument("--target-random-seed", type=int, default=42850)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--margin-positive-threshold", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true", help="Use smaller Windows-friendly budgets.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    run_dir = Path(args.run_dir)
    clusters = _parse_clusters(args.clusters)
    known_label_dirs = [Path(path) for path in (args.label_dir or DEFAULT_LABEL_DIRS)]
    produced_dirs: list[Path] = []
    command_log: list[list[str]] = []

    pairs = min(int(args.pairs), 8) if args.quick else int(args.pairs)
    target_states = min(int(args.target_states), 32) if args.quick else int(args.target_states)
    targets_per_pair = min(int(args.targets_per_pair), 3) if args.quick else int(args.targets_per_pair)
    actions_per_state = min(int(args.actions_per_state), 4) if args.quick else int(args.actions_per_state)
    max_forced_replays = min(int(args.max_forced_replays), 160) if args.quick else int(args.max_forced_replays)
    stop_after = min(int(args.stop_after_positives_per_cluster), 1) if args.quick else int(args.stop_after_positives_per_cluster)

    for cluster_index, cluster in enumerate(clusters):
        artifact = f"b1_cf_labels_s1_big_{args.tag}_{cluster.name}"
        output_dir = run_dir / "eval" / artifact
        exclude_paths = [path / "counterfactual_labels.jsonl" for path in [*known_label_dirs, *produced_dirs]]
        cmd = [
            sys.executable,
            _script_path("b1_counterfactual_labels.py"),
            "--stack-config",
            args.stack_config,
            "--run-dir",
            args.run_dir,
            "--b1-baseline-run-dir",
            args.b1_baseline_run_dir,
            "--checkpoint-policy",
            args.checkpoint_policy,
            "--pairs",
            str(pairs),
            "--artifact-dir-name",
            artifact,
            "--seed-scope",
            f"b1_cf_labels_s1_big_{args.tag}_{cluster.name}",
            "--device",
            args.device,
            "--public-heuristic-bias-scale",
            "1.0",
            "--max-target-states",
            str(target_states),
            "--max-targets-per-pair",
            str(targets_per_pair),
            "--max-actions-per-state",
            str(actions_per_state),
            "--family-representatives-per-family",
            str(cluster.family_representatives_per_family),
            "--max-forced-replays",
            str(max_forced_replays),
            "--progress-every",
            str(args.progress_every),
            "--stop-after-positive-labels",
            str(stop_after),
            "--randomize-target-order",
            "--target-random-seed",
            str(int(args.target_random_seed) + cluster_index),
            "--target-families",
            cluster.target_families,
            "--execution-mode",
            "in_process",
            "--ignore-excluded-label-pair-indices",
            "--margin-positive-threshold",
            str(float(args.margin_positive_threshold)),
        ]
        if cluster.require_pass_legal:
            cmd.append("--require-pass-legal")
        if cluster.require_baseline_family:
            cmd.extend(["--require-baseline-family", cluster.require_baseline_family])
        if cluster.exclude_candidate_family:
            cmd.extend(["--exclude-candidate-family", cluster.exclude_candidate_family])
        if cluster.exclude_candidate_action_id:
            cmd.extend(["--exclude-candidate-action-id", cluster.exclude_candidate_action_id])
        if cluster.two_step_beam_targets > 0:
            cmd.extend(["--two-step-beam-targets", str(cluster.two_step_beam_targets)])
            cmd.extend(["--two-step-second-actions", str(cluster.two_step_second_actions)])
            cmd.extend(["--two-step-min-first-delta", str(cluster.two_step_min_first_delta)])
            if cluster.two_step_include_positive_first:
                cmd.append("--two-step-include-positive-first")
        _append_repeated(cmd, "--exclude-labels", [str(path) for path in exclude_paths])
        command_log.append(cmd)
        print("\n$ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(cwd), check=True)
            if _read_jsonl(output_dir / "counterfactual_labels.jsonl"):
                produced_dirs.append(output_dir)

    all_dirs = [*known_label_dirs, *produced_dirs]
    summary = {
        "format": "b1_big_label_miner_summary_v1",
        "tag": args.tag,
        "dry_run": bool(args.dry_run),
        "clusters": [cluster.name for cluster in clusters],
        "quick": bool(args.quick),
        "produced_label_dirs": [str(path) for path in produced_dirs],
        "commands": command_log,
        "aggregate": _summarize_labels(all_dirs),
    }
    summary_dir = run_dir / "eval" / f"b1_big_label_miner_{args.tag}"
    _write_json(summary_dir / "big_label_miner_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

