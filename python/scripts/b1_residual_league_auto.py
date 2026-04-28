"""Automatic frozen-B1 residual league runner.

This script turns the manual rescue loop into a repeatable server-side loop:

1. train a warmstarted main residual for a short block;
2. extract the residual head from the checkpoint;
3. evaluate the extracted residual against B1 S1 in both directions;
4. promote strong extracted residuals into the next block's hard-negative lane;
5. write a league-state JSON artifact after every iteration.

It deliberately promotes extracted residual states, not full checkpoints.  The
base B1 remains frozen and the evolving object is the residual deviation head.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from weiss_rl.config import load_stack_config


DEFAULT_BASE_STACK_CONFIG = "configs/residual_league_s1_server.yaml"
DEFAULT_EVAL_STACK_CONFIG = "configs/residual_eval_s1.yaml"
DEFAULT_RUN_DIR = "runs/b1_s1_distillonly_u450_to_u455_20260427"
DEFAULT_B1_BASELINE_RUN_DIR = "runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425"
DEFAULT_B1_CHECKPOINT = (
    "runs/b1_continue_u100_trainheurrows_lowlr_u450_s3_20260425/"
    "training/checkpoints/checkpoint_450.pt"
)
DEFAULT_INITIAL_RESIDUAL = (
    "runs/b1_s1_distillonly_u450_to_u455_20260427/"
    "eval/b1_residual_adoption_probe_rush_trainlive_mine1_20260428a/residual_state.pt"
)
DEFAULT_LABEL_DIR = (
    "runs/b1_s1_distillonly_u450_to_u455_20260427/"
    "eval/b1_cf_labels_s1_candidate_reps_p8_t24_a6_tensor_20260427"
)


@dataclass(frozen=True, slots=True)
class ResidualOpponent:
    policy_id: str
    role: str
    base_snapshot_path: str
    residual_state_path: str
    public_heuristic_bias_scale: float = 1.0


def _script_path(name: str) -> str:
    return str(Path("python/scripts") / name)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _run(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(str(item) for item in cmd), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in cmd], cwd=str(cwd), check=True)


def _default_tag() -> str:
    return datetime.now().strftime("resleague_%Y%m%d_%H%M%S")


def _config_opponents(config_path: Path) -> list[ResidualOpponent]:
    stack = load_stack_config(config_path)
    opponents: list[ResidualOpponent] = []
    for item in stack.config.training.residual_opponent_policies:
        opponents.append(
            ResidualOpponent(
                policy_id=str(item.policy_id),
                role=str(item.role),
                base_snapshot_path=str(item.base_snapshot_path),
                residual_state_path=str(item.residual_state_path),
                public_heuristic_bias_scale=float(item.public_heuristic_bias_scale),
            )
        )
    return opponents


def _score_report(report: Mapping[str, Any], *, residual_as_opponent: bool) -> dict[str, Any]:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    pair_summary = report.get("pair_class_summary", {}) if isinstance(report.get("pair_class_summary"), Mapping) else {}
    uncertainty = report.get("uncertainty", {}) if isinstance(report.get("uncertainty"), Mapping) else {}
    mean = float(uncertainty.get("mean", 0.0))
    residual_mean = 1.0 - mean if residual_as_opponent else mean
    pair_counts = pair_summary.get("pair_class_counts", {})
    if not isinstance(pair_counts, Mapping):
        pair_counts = {}
    if residual_as_opponent:
        residual_2_0 = int(pair_counts.get("0-2", 0))
        residual_0_2 = int(pair_counts.get("2-0", 0))
    else:
        residual_2_0 = int(pair_counts.get("2-0", 0))
        residual_0_2 = int(pair_counts.get("0-2", 0))
    return {
        "mean_reported": mean,
        "residual_mean": residual_mean,
        "wins_reported": int(summary.get("wins", 0)),
        "losses_reported": int(summary.get("losses", 0)),
        "pair_class_counts": dict(pair_counts),
        "residual_2_0_pairs": residual_2_0,
        "residual_0_2_pairs": residual_0_2,
        "pair_score_mean_reported": pair_summary.get("pair_score_mean"),
        "residual_family_drift_rate": (report.get("residual_trace_drift", {}) or {}).get(
            "selected_family_differs_from_final_top1_rate"
        ),
    }


def _load_closed_loop_report(run_dir: Path, artifact_name: str) -> dict[str, Any]:
    return dict(_read_json(run_dir / "eval" / artifact_name / "closed_loop_report.json"))


def _promotion_passes(
    *,
    focal: Mapping[str, Any],
    reverse: Mapping[str, Any] | None,
    promote_threshold: float,
    require_positive_pair_score: bool,
) -> bool:
    if float(focal.get("residual_mean", 0.0)) < float(promote_threshold):
        return False
    if require_positive_pair_score and int(focal.get("residual_2_0_pairs", 0)) <= int(focal.get("residual_0_2_pairs", 0)):
        return False
    if reverse is not None:
        if float(reverse.get("residual_mean", 0.0)) < float(promote_threshold):
            return False
        if require_positive_pair_score and int(reverse.get("residual_2_0_pairs", 0)) <= int(reverse.get("residual_0_2_pairs", 0)):
            return False
    return True


def _generated_config_payload(
    *,
    base_stack_config: Path,
    base_snapshot_path: str,
    initial_residual_state_path: str,
    opponents: Sequence[ResidualOpponent],
    opponent_ids: Sequence[str],
    alpha: float,
    hidden_dim: int,
    residual_mode: str,
    gate_bias: float,
    learning_rate: float,
    counterfactual_coef: float,
    counterfactual_final_coef: float,
    counterfactual_end_updates: int,
    counterfactual_max_labels: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "description": "Generated residual-league iteration config; do not edit by hand.",
        "extends": str(base_stack_config.resolve()),
        "training": {
            "diverse_opponent_actor_count": -1,
            "diverse_model_actor_count": 1,
            "diverse_opponent_policy_ids": list(opponent_ids),
            "main_residual_policy": {
                "enabled": True,
                "base_snapshot_path": str(base_snapshot_path),
                "initial_residual_state_path": str(initial_residual_state_path),
                "public_heuristic_bias_scale": 1.0,
                "hidden_dim": int(hidden_dim),
                "alpha": float(alpha),
                "residual_mode": str(residual_mode),
                "gate_bias": float(gate_bias),
            },
            "optimizer": {"learning_rate": float(learning_rate)},
            "counterfactual_positive": {
                "coef": float(counterfactual_coef),
                "final_coef": float(counterfactual_final_coef),
                "start_updates": 0,
                "end_updates": int(counterfactual_end_updates),
                "max_labels": int(counterfactual_max_labels),
            },
            "residual_opponent_policies": [asdict(item) for item in opponents],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=_default_tag())
    parser.add_argument("--base-stack-config", default=DEFAULT_BASE_STACK_CONFIG)
    parser.add_argument("--eval-stack-config", default=DEFAULT_EVAL_STACK_CONFIG)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--b1-baseline-run-dir", default=DEFAULT_B1_BASELINE_RUN_DIR)
    parser.add_argument("--base-snapshot-path", default=DEFAULT_B1_CHECKPOINT)
    parser.add_argument("--initial-residual-state", default=DEFAULT_INITIAL_RESIDUAL)
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--updates-per-iteration", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--unroll-length", type=int, default=8)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=1)
    parser.add_argument("--screen-pairs", type=int, default=16)
    parser.add_argument("--confirm-pairs", type=int, default=32)
    parser.add_argument("--promote-threshold", type=float, default=0.75)
    parser.add_argument("--confirm-threshold", type=float, default=0.875)
    parser.add_argument("--require-positive-pair-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-reverse-eval", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-mode", default="plain")
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--counterfactual-coef", type=float, default=3.0)
    parser.add_argument("--counterfactual-final-coef", type=float, default=2.0)
    parser.add_argument("--counterfactual-end-updates", type=int, default=20)
    parser.add_argument("--counterfactual-max-labels", type=int, default=10)
    parser.add_argument("--max-opponents", type=int, default=12)
    parser.add_argument(
        "--continue-policy",
        choices=("latest", "best"),
        default="latest",
        help="Which extracted residual initializes the next iteration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    base_stack_config = Path(args.base_stack_config)
    eval_stack_config = Path(args.eval_stack_config)
    run_dir = Path(args.run_dir)
    league_dir = run_dir / "eval" / f"b1_residual_league_auto_{args.tag}"
    # Do not name this directory "configs": the stack loader resolves repo root
    # by walking upward to the first parent containing a configs/ directory.
    generated_config_dir = league_dir / "generated_presets"
    extracted_dir = league_dir / "extracted"

    opponents = _config_opponents(base_stack_config)
    opponent_ids = [item.policy_id for item in opponents]
    current_state = str(args.initial_residual_state)
    best_state = current_state
    best_score = 0.0
    iterations: list[dict[str, Any]] = []

    for index in range(1, int(args.iterations) + 1):
        iter_tag = f"{args.tag}_iter{index:02d}"
        config_path = generated_config_dir / f"{iter_tag}.yaml"
        active_opponents = opponents[-int(args.max_opponents) :] if int(args.max_opponents) > 0 else opponents
        active_ids = [item.policy_id for item in active_opponents if item.policy_id in set(opponent_ids)]
        payload = _generated_config_payload(
            base_stack_config=base_stack_config,
            base_snapshot_path=str(args.base_snapshot_path),
            initial_residual_state_path=current_state,
            opponents=active_opponents,
            opponent_ids=active_ids,
            alpha=float(args.alpha),
            hidden_dim=int(args.hidden_dim),
            residual_mode=str(args.residual_mode),
            gate_bias=float(args.gate_bias),
            learning_rate=float(args.learning_rate),
            counterfactual_coef=float(args.counterfactual_coef),
            counterfactual_final_coef=float(args.counterfactual_final_coef),
            counterfactual_end_updates=int(args.counterfactual_end_updates),
            counterfactual_max_labels=int(args.counterfactual_max_labels),
        )
        _write_yaml(config_path, payload)

        train_label = f"b1_residual_league_{iter_tag}"
        train_run = Path("runs") / train_label
        train_cmd = [
            sys.executable,
            _script_path("train.py"),
            "--stack-config",
            str(config_path),
            "--run-label",
            train_label,
            "--max-updates",
            str(int(args.updates_per_iteration)),
            "--num-envs",
            str(int(args.num_envs)),
            "--unroll-length",
            str(int(args.unroll_length)),
            "--checkpoint-interval-updates",
            str(int(args.checkpoint_interval_updates)),
            "--device",
            args.device,
            "--runtime-mode",
            "train_ordered",
            "--override",
            "system.collection_backend=\"auto\"",
            "--b1-baseline-run-dir",
            args.b1_baseline_run_dir,
        ]
        _run(train_cmd, cwd=cwd, dry_run=bool(args.dry_run))

        checkpoint = train_run / "training" / "checkpoints" / f"checkpoint_{int(args.updates_per_iteration)}.pt"
        residual_state = extracted_dir / f"{iter_tag}" / "residual_state.pt"
        extract_cmd = [
            sys.executable,
            _script_path("b1_extract_main_residual_state.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(residual_state),
            "--alpha",
            str(float(args.alpha)),
            "--residual-mode",
            str(args.residual_mode),
            "--gate-bias",
            str(float(args.gate_bias)),
        ]
        _run(extract_cmd, cwd=cwd, dry_run=bool(args.dry_run))

        screen_artifact = f"b1_residual_league_{iter_tag}_screen_focal_p{int(args.screen_pairs)}"
        eval_pairs = int(args.screen_pairs)
        eval_cmd = [
            sys.executable,
            _script_path("b1_residual_closed_loop_eval.py"),
            "--stack-config",
            str(eval_stack_config),
            "--run-dir",
            str(run_dir),
            "--b1-baseline-run-dir",
            str(args.b1_baseline_run_dir),
            "--label-dir",
            str(args.label_dir),
            "--residual-state",
            str(residual_state),
            "--pairs",
            str(eval_pairs),
            "--artifact-dir-name",
            screen_artifact,
            "--seed-scope",
            screen_artifact,
            "--device",
            args.device,
            "--public-heuristic-bias-scale",
            "1.0",
            "--action-rng-salt-mode",
            "physical",
        ]
        _run(eval_cmd, cwd=cwd, dry_run=bool(args.dry_run))
        screen_score: dict[str, Any] = {}
        focal_score: dict[str, Any] = {}
        reverse_score: dict[str, Any] | None = None
        if not args.dry_run:
            screen_report = _load_closed_loop_report(run_dir, screen_artifact)
            screen_score = _score_report(screen_report, residual_as_opponent=False)

        focal_artifact = ""
        reverse_artifact = ""
        should_confirm = bool(args.dry_run) or float(screen_score.get("residual_mean", 0.0)) >= float(args.promote_threshold)
        if should_confirm:
            focal_artifact = f"b1_residual_league_{iter_tag}_confirm_focal_p{int(args.confirm_pairs)}"
            confirm_cmd = [
                sys.executable,
                _script_path("b1_residual_closed_loop_eval.py"),
                "--stack-config",
                str(eval_stack_config),
                "--run-dir",
                str(run_dir),
                "--b1-baseline-run-dir",
                str(args.b1_baseline_run_dir),
                "--label-dir",
                str(args.label_dir),
                "--residual-state",
                str(residual_state),
                "--pairs",
                str(int(args.confirm_pairs)),
                "--artifact-dir-name",
                focal_artifact,
                "--seed-scope",
                focal_artifact,
                "--device",
                args.device,
                "--public-heuristic-bias-scale",
                "1.0",
                "--action-rng-salt-mode",
                "physical",
            ]
            _run(confirm_cmd, cwd=cwd, dry_run=bool(args.dry_run))
            if not args.dry_run:
                focal_report = _load_closed_loop_report(run_dir, focal_artifact)
                focal_score = _score_report(focal_report, residual_as_opponent=False)

            if not bool(args.skip_reverse_eval):
                reverse_artifact = f"b1_residual_league_{iter_tag}_confirm_asopp_p{int(args.confirm_pairs)}"
                reverse_cmd = [
                    sys.executable,
                    _script_path("b1_residual_closed_loop_eval.py"),
                    "--stack-config",
                    str(eval_stack_config),
                    "--run-dir",
                    str(run_dir),
                    "--b1-baseline-run-dir",
                    str(args.b1_baseline_run_dir),
                    "--label-dir",
                    str(args.label_dir),
                    "--residual-state",
                    str(residual_state),
                    "--pairs",
                    str(int(args.confirm_pairs)),
                    "--artifact-dir-name",
                    reverse_artifact,
                    "--seed-scope",
                    reverse_artifact,
                    "--device",
                    args.device,
                    "--public-heuristic-bias-scale",
                    "1.0",
                    "--action-rng-salt-mode",
                    "physical",
                    "--residual-as-opponent",
                ]
                _run(reverse_cmd, cwd=cwd, dry_run=bool(args.dry_run))
                if not args.dry_run:
                    reverse_report = _load_closed_loop_report(run_dir, reverse_artifact)
                    reverse_score = _score_report(reverse_report, residual_as_opponent=True)
        else:
            focal_score = screen_score

        promote = False
        confirm = False
        if not args.dry_run and should_confirm:
            promote = _promotion_passes(
                focal=focal_score,
                reverse=reverse_score,
                promote_threshold=float(args.promote_threshold),
                require_positive_pair_score=bool(args.require_positive_pair_score),
            )
            confirm = _promotion_passes(
                focal=focal_score,
                reverse=reverse_score,
                promote_threshold=float(args.confirm_threshold),
                require_positive_pair_score=bool(args.require_positive_pair_score),
            )
        if not args.dry_run:
            residual_mean = float((focal_score or screen_score).get("residual_mean", 0.0))
            if reverse_score is not None:
                residual_mean = min(residual_mean, float(reverse_score.get("residual_mean", 0.0)))
            if residual_mean > best_score:
                best_score = residual_mean
                best_state = str(residual_state)
            if promote:
                policy_id = f"b1_residual_auto_{args.tag}_iter{index:02d}"
                role = "b1_residual_auto_confirmed_hard_negative" if confirm else "b1_residual_auto_provisional_hard_negative"
                opponents.append(
                    ResidualOpponent(
                        policy_id=policy_id,
                        role=role,
                        base_snapshot_path=str(args.base_snapshot_path),
                        residual_state_path=str(residual_state),
                        public_heuristic_bias_scale=1.0,
                    )
                )
                opponent_ids.append(policy_id)
            current_state = str(residual_state) if str(args.continue_policy) == "latest" else best_state

        iteration_payload = {
            "iteration": index,
            "train_run": str(train_run),
            "config_path": str(config_path),
            "checkpoint": str(checkpoint),
            "extracted_residual_state": str(residual_state),
            "screen_artifact": screen_artifact,
            "screen": screen_score,
            "focal_artifact": focal_artifact,
            "reverse_artifact": reverse_artifact,
            "focal": focal_score,
            "reverse": reverse_score,
            "promoted": promote,
            "confirmed": confirm,
            "next_initial_residual_state": current_state,
            "opponent_count_next": len(opponents),
        }
        iterations.append(iteration_payload)
        _write_json(
            league_dir / "residual_league_state.json",
            {
                "format": "b1_residual_league_auto_state_v1",
                "tag": args.tag,
                "dry_run": bool(args.dry_run),
                "iterations": iterations,
                "opponents": [asdict(item) for item in opponents],
                "best_state": best_state,
                "best_score": best_score,
            },
        )

    final_state = {
        "format": "b1_residual_league_auto_state_v1",
        "tag": args.tag,
        "dry_run": bool(args.dry_run),
        "iterations": iterations,
        "opponents": [asdict(item) for item in opponents],
        "best_state": best_state,
        "best_score": best_score,
    }
    _write_json(league_dir / "residual_league_state.json", final_state)
    print(json.dumps(final_state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

