"""Run a common-public-bias sweep using b1_artifact_matrix.py.

This wrapper is diagnostic glue: it repeatedly invokes the artifact matrix at
fixed public heuristic bias scales, then aggregates the pair-class outcomes into
a compact JSON table. It does not change training or promotion behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_DEFAULT_MATCHUPS = (
    "B1=u480",
    "u480=B1",
    "B1=u540",
    "u540=B1",
    "u480=u540",
    "u540=u480",
)


def _parse_scales(raw: str) -> list[float]:
    scales: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if value < 0.0:
            raise argparse.ArgumentTypeError("bias scales must be non-negative")
        scales.append(value)
    if not scales:
        raise argparse.ArgumentTypeError("provide at least one scale")
    return scales


def _scale_slug(scale: float) -> str:
    text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    return "s" + text.replace(".", "p")


def _matrix_summary_path(run_dir: Path, artifact_dir_name: str) -> Path:
    return run_dir / "eval" / artifact_dir_name / "matrix_summary.json"


def _run_matrix(
    *,
    python_exe: str,
    matrix_script: Path,
    stack_config: str,
    run_dir: Path,
    b1_baseline_run_dir: Path,
    checkpoint_policies: list[str],
    pairs: int,
    artifact_dir_name: str,
    device: str,
    scale: float,
    matchups: list[str],
    extra_args: list[str],
) -> None:
    command = [
        python_exe,
        str(matrix_script),
        "--stack-config",
        stack_config,
        "--run-dir",
        str(run_dir),
        "--b1-baseline-run-dir",
        str(b1_baseline_run_dir),
        "--pairs",
        str(int(pairs)),
        "--artifact-dir-name",
        artifact_dir_name,
        "--device",
        str(device),
        "--public-heuristic-bias-scale",
        str(float(scale)),
    ]
    for policy in checkpoint_policies:
        command.extend(["--checkpoint-policy", policy])
    for matchup in matchups:
        command.extend(["--matchup", matchup])
    command.extend(extra_args)
    subprocess.run(command, check=True)


def _aggregate_scale_summary(
    *,
    scale: float,
    artifact_dir_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for matchup, payload in sorted(summary.get("matchups", {}).items()):
        pair_counts = payload.get("pair_class_summary", {}).get("pair_class_counts", {})
        counters = payload.get("matrix_runner_counters", {})
        rows.append(
            {
                "scale": float(scale),
                "artifact_dir_name": artifact_dir_name,
                "matchup": matchup,
                "mean": payload.get("mean"),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                "draws": payload.get("draws"),
                "truncations": payload.get("truncations"),
                "pair_2_0": pair_counts.get("2-0", 0),
                "pair_1_1": pair_counts.get("1-1", 0),
                "pair_0_2": pair_counts.get("0-2", 0),
                "pair_mixed": pair_counts.get("mixed", 0),
                "model_decisions": counters.get("model_decisions"),
                "sample_decisions": counters.get("sample_decisions"),
                "greedy_override_decisions": counters.get("greedy_override_decisions"),
                "heuristic_decisions": counters.get("heuristic_decisions"),
                "random_legal_decisions": counters.get("random_legal_decisions"),
            }
        )
    return {
        "scale": float(scale),
        "artifact_dir_name": artifact_dir_name,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-policy", action="append", default=[])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--artifact-dir-name", default="b1_bias_sweep_matrix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scales", type=_parse_scales, default=_parse_scales("0.0,0.5,1.0,1.5,2.0,3.0"))
    parser.add_argument("--matchup", action="append", default=[])
    parser.add_argument("--matrix-extra-arg", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_policy:
        raise SystemExit("provide at least one --checkpoint-policy alias=checkpoint.pt")
    if int(args.pairs) <= 0:
        raise SystemExit("--pairs must be positive")

    script_path = Path(__file__).with_name("b1_artifact_matrix.py")
    matchups = list(args.matchup or _DEFAULT_MATCHUPS)
    sweep_root = args.run_dir / "eval" / args.artifact_dir_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    scale_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for scale in args.scales:
        scale_artifact = f"{args.artifact_dir_name}_{_scale_slug(scale)}"
        summary_path = _matrix_summary_path(args.run_dir, scale_artifact)
        if not args.skip_existing or not summary_path.is_file():
            _run_matrix(
                python_exe=sys.executable,
                matrix_script=script_path,
                stack_config=str(args.stack_config),
                run_dir=args.run_dir,
                b1_baseline_run_dir=args.b1_baseline_run_dir,
                checkpoint_policies=list(args.checkpoint_policy),
                pairs=int(args.pairs),
                artifact_dir_name=scale_artifact,
                device=str(args.device),
                scale=float(scale),
                matchups=matchups,
                extra_args=list(args.matrix_extra_arg),
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scale_summary = _aggregate_scale_summary(
            scale=float(scale),
            artifact_dir_name=scale_artifact,
            summary=summary,
        )
        scale_summaries.append(scale_summary)
        rows.extend(scale_summary["rows"])

    payload = {
        "format": "b1_bias_sweep_matrix_v1",
        "run_dir": args.run_dir.as_posix(),
        "stack_config": str(args.stack_config),
        "pairs": int(args.pairs),
        "scales": [float(scale) for scale in args.scales],
        "matchups": matchups,
        "scale_summaries": scale_summaries,
        "rows": rows,
    }
    out_path = sweep_root / "bias_sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote bias sweep summary to {out_path}")
    for row in rows:
        print(
            f"scale={row['scale']:.3g} {row['matchup']}: "
            f"mean={row['mean']} pair_classes="
            f"2-0:{row['pair_2_0']} 1-1:{row['pair_1_1']} 0-2:{row['pair_0_2']}"
        )


if __name__ == "__main__":
    main()
