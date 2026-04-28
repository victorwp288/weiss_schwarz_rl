"""Fast funnel for B1 S1 counterfactual-label residual experiments.

This wrapper is intentionally orchestration-only. It uses the existing
counterfactual mining, residual adoption, and closed-loop eval scripts, then
writes a compact summary that makes it easy to decide whether a candidate earns
more expensive confirmation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def _script_path(name: str) -> str:
    return str(Path("python/scripts") / name)


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _append_repeated(cmd: list[str], flag: str, values: Iterable[str]) -> None:
    for value in values:
        cmd.extend([flag, value])


def _counterfactual_artifact_dir(run_dir: Path, artifact_name: str) -> Path:
    return run_dir / "eval" / artifact_name


def _count_labels(label_dirs: Sequence[Path]) -> int:
    total = 0
    for label_dir in label_dirs:
        total += len(_read_jsonl(label_dir / "counterfactual_labels.jsonl"))
    return total


def _load_closed_loop_report(run_dir: Path, artifact_name: str) -> dict[str, Any]:
    path = run_dir / "eval" / artifact_name / "closed_loop_report.json"
    return dict(_read_json(path))


def _score_report(report: Mapping[str, Any], *, residual_as_opponent: bool) -> dict[str, Any]:
    summary = report.get("summary", {})
    pair_summary = report.get("pair_class_summary", {})
    uncertainty = report.get("uncertainty", {})
    mean = float(uncertainty.get("mean", 0.0))
    residual_mean = 1.0 - mean if residual_as_opponent else mean
    return {
        "mean_reported": mean,
        "residual_mean": residual_mean,
        "wins_reported": int(summary.get("wins", 0)),
        "losses_reported": int(summary.get("losses", 0)),
        "pair_class_counts": pair_summary.get("pair_class_counts", {}),
        "pair_score_mean_reported": pair_summary.get("pair_score_mean"),
        "residual_family_drift_rate": (report.get("residual_trace_drift", {}) or {}).get(
            "selected_family_differs_from_final_top1_rate"
        ),
    }


def _default_tag() -> str:
    return datetime.now().strftime("rush_%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=_default_tag())
    parser.add_argument("--stack-config", default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--b1-baseline-run-dir", default=DEFAULT_B1_BASELINE_RUN_DIR)
    parser.add_argument("--checkpoint-policy", default=f"u450={DEFAULT_B1_CHECKPOINT}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label-dir", action="append", default=[])
    parser.add_argument("--skip-mining", action="store_true")
    parser.add_argument("--skip-reverse-eval", action="store_true")
    parser.add_argument("--confirm-if-screen-ge", type=float, default=0.58)
    parser.add_argument("--screen-pairs", type=int, default=8)
    parser.add_argument("--confirm-pairs", type=int, default=16)
    parser.add_argument("--mine-pairs", type=int, default=8)
    parser.add_argument("--mine-target-states", type=int, default=32)
    parser.add_argument("--mine-targets-per-pair", type=int, default=4)
    parser.add_argument("--mine-actions-per-state", type=int, default=4)
    parser.add_argument("--mine-max-forced-replays", type=int, default=180)
    parser.add_argument("--mine-stop-after-positives", type=int, default=1)
    parser.add_argument("--mine-target-seed", type=int, default=42802)
    parser.add_argument("--mine-target-families", default="main_play_character")
    parser.add_argument("--require-pass-legal", action="store_true", default=True)
    parser.add_argument("--require-baseline-family", default="main_play_character")
    parser.add_argument("--train-steps", type=int, default=220)
    parser.add_argument("--final-train-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--residual-l2-coef", type=float, default=0.002)
    parser.add_argument("--residual-mode", choices=("plain", "gated", "family_gated"), default="plain")
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument(
        "--trainer",
        choices=("stored-logit", "trainable-live"),
        default="stored-logit",
        help="Residual trainer to use before closed-loop eval.",
    )
    parser.add_argument("--early-stop-adoption-rate", type=float, default=0.0)
    parser.add_argument("--early-stop-mean-positive-prob", type=float, default=0.0)
    parser.add_argument("--early-stop-check-every", type=int, default=25)
    parser.add_argument("--rerun-final-if-screen-passes", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    run_dir = Path(args.run_dir)
    label_dirs = [Path(path) for path in (args.label_dir or DEFAULT_LABEL_DIRS)]
    tag = args.tag
    summary_path = run_dir / "eval" / f"b1_rush_residual_loop_{tag}" / "rush_summary.json"

    mined_dir: Path | None = None
    if not args.skip_mining:
        mine_artifact = f"b1_cf_labels_s1_rush_{tag}"
        mine_cmd = [
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
            str(args.mine_pairs),
            "--artifact-dir-name",
            mine_artifact,
            "--seed-scope",
            f"b1_cf_labels_s1_rush_{tag}",
            "--device",
            args.device,
            "--public-heuristic-bias-scale",
            "1.0",
            "--max-target-states",
            str(args.mine_target_states),
            "--max-targets-per-pair",
            str(args.mine_targets_per_pair),
            "--max-actions-per-state",
            str(args.mine_actions_per_state),
            "--family-representatives-per-family",
            "1",
            "--max-forced-replays",
            str(args.mine_max_forced_replays),
            "--progress-every",
            str(args.progress_every),
            "--stop-after-positive-labels",
            str(args.mine_stop_after_positives),
            "--randomize-target-order",
            "--target-random-seed",
            str(args.mine_target_seed),
            "--target-families",
            args.mine_target_families,
            "--execution-mode",
            "in_process",
            "--ignore-excluded-label-pair-indices",
        ]
        if args.require_pass_legal:
            mine_cmd.append("--require-pass-legal")
        if args.require_baseline_family:
            mine_cmd.extend(["--require-baseline-family", args.require_baseline_family])
        _append_repeated(mine_cmd, "--exclude-labels", [str(path / "counterfactual_labels.jsonl") for path in label_dirs])
        _run(mine_cmd, cwd=cwd)
        mined_dir = _counterfactual_artifact_dir(run_dir, mine_artifact)
        if (mined_dir / "counterfactual_labels.jsonl").is_file() and _read_jsonl(
            mined_dir / "counterfactual_labels.jsonl"
        ):
            label_dirs.append(mined_dir)

    label_count = _count_labels(label_dirs)
    max_labels = args.max_labels if args.max_labels > 0 else label_count
    adoption_artifact = f"b1_residual_adoption_probe_rush_{tag}"
    adoption_dir = run_dir / "eval" / adoption_artifact

    def train_residual(steps: int, output_dir: Path) -> dict[str, Any]:
        script_name = (
            "b1_trainable_residual_policy.py" if str(args.trainer) == "trainable-live" else "b1_residual_adoption_probe.py"
        )
        train_cmd = [
            sys.executable,
            _script_path(script_name),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--steps",
            str(steps),
            "--lr",
            str(args.lr),
            "--alpha",
            str(args.alpha),
            "--hidden-dim",
            str(args.hidden_dim),
            "--margin",
            str(args.margin),
            "--residual-l2-coef",
            str(args.residual_l2_coef),
            "--residual-mode",
            args.residual_mode,
            "--gate-bias",
            str(args.gate_bias),
            "--validation-fraction",
            str(args.validation_fraction),
            "--max-labels",
            str(max_labels),
        ]
        if str(args.trainer) == "trainable-live":
            train_cmd[2:2] = [
                "--stack-config",
                args.stack_config,
                "--run-dir",
                args.run_dir,
                "--b1-baseline-run-dir",
                args.b1_baseline_run_dir,
            ]
            if float(args.early_stop_adoption_rate) > 0.0:
                train_cmd.extend(
                    [
                        "--early-stop-adoption-rate",
                        str(args.early_stop_adoption_rate),
                        "--early-stop-mean-positive-prob",
                        str(args.early_stop_mean_positive_prob),
                        "--early-stop-check-every",
                        str(args.early_stop_check_every),
                    ]
                )
        _append_repeated(train_cmd, "--label-dir", [str(path) for path in label_dirs])
        _run(train_cmd, cwd=cwd)
        report_name = "trainable_residual_report.json" if str(args.trainer) == "trainable-live" else "residual_adoption_report.json"
        return dict(_read_json(output_dir / report_name))

    adoption_report = train_residual(args.train_steps, adoption_dir)
    residual_state = adoption_dir / "residual_state.pt"
    first_label_dir = label_dirs[0]

    def run_eval(*, pairs: int, artifact: str, residual_as_opponent: bool) -> dict[str, Any]:
        eval_cmd = [
            sys.executable,
            _script_path("b1_residual_closed_loop_eval.py"),
            "--stack-config",
            args.stack_config,
            "--run-dir",
            args.run_dir,
            "--b1-baseline-run-dir",
            args.b1_baseline_run_dir,
            "--label-dir",
            str(first_label_dir),
            "--residual-state",
            str(residual_state),
            "--pairs",
            str(pairs),
            "--artifact-dir-name",
            artifact,
            "--seed-scope",
            artifact,
            "--device",
            args.device,
            "--public-heuristic-bias-scale",
            "1.0",
            "--action-rng-salt-mode",
            "physical",
        ]
        if residual_as_opponent:
            eval_cmd.append("--residual-as-opponent")
        _run(eval_cmd, cwd=cwd)
        return _load_closed_loop_report(run_dir, artifact)

    screen_artifact = f"b1_residual_closed_loop_rush_{tag}_focal_p{args.screen_pairs}"
    screen_report = run_eval(pairs=args.screen_pairs, artifact=screen_artifact, residual_as_opponent=False)
    screen_score = _score_report(screen_report, residual_as_opponent=False)

    final_adoption_report = adoption_report
    if args.rerun_final_if_screen_passes and float(screen_score["residual_mean"]) >= args.confirm_if_screen_ge:
        final_dir = run_dir / "eval" / f"b1_residual_adoption_probe_rush_{tag}_final{args.final_train_steps}"
        final_adoption_report = train_residual(args.final_train_steps, final_dir)
        residual_state = final_dir / "residual_state.pt"
        adoption_dir = final_dir

    confirm_reports: dict[str, Any] = {}
    if float(screen_score["residual_mean"]) >= args.confirm_if_screen_ge:
        focal_artifact = f"b1_residual_closed_loop_rush_{tag}_confirm_focal_p{args.confirm_pairs}"
        focal_report = run_eval(pairs=args.confirm_pairs, artifact=focal_artifact, residual_as_opponent=False)
        confirm_reports["focal"] = _score_report(focal_report, residual_as_opponent=False)
        if not args.skip_reverse_eval:
            reverse_artifact = f"b1_residual_closed_loop_rush_{tag}_confirm_asopp_p{args.confirm_pairs}"
            reverse_report = run_eval(pairs=args.confirm_pairs, artifact=reverse_artifact, residual_as_opponent=True)
            confirm_reports["as_opponent"] = _score_report(reverse_report, residual_as_opponent=True)

    payload = {
        "format": "b1_rush_residual_loop_summary_v1",
        "tag": tag,
        "device": args.device,
        "trainer": str(args.trainer),
        "mined_label_dir": str(mined_dir) if mined_dir is not None else None,
        "label_dirs": [str(path) for path in label_dirs],
        "label_count": label_count,
        "max_labels": max_labels,
        "adoption_dir": str(adoption_dir),
        "adoption": {
            "adoption_rate": final_adoption_report.get("adoption_rate"),
            "adoption_count": final_adoption_report.get("adoption_count"),
            "label_count": final_adoption_report.get("label_count"),
            "base_param_delta_l2": final_adoption_report.get("base_param_delta_l2"),
            "residual_zero_logit_max_abs_diff_vs_stored_b1_s1": final_adoption_report.get(
                "residual_zero_logit_max_abs_diff_vs_stored_b1_s1"
            )
            or final_adoption_report.get("live_identity_max_abs_diff_vs_stored_b1_s1_at_init"),
            "train_loss_initial": final_adoption_report.get("train_loss_initial"),
            "train_loss_final": final_adoption_report.get("train_loss_final"),
            "stopped_at_step": final_adoption_report.get("stopped_at_step"),
        },
        "screen": screen_score,
        "confirm": confirm_reports,
        "promoted_to_confirm": bool(confirm_reports),
        "decision_hint": (
            "confirm_or_integrate_candidate"
            if confirm_reports
            else "reject_or_retrain_candidate_before_confirm"
        ),
    }
    _write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

