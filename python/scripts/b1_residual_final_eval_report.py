"""Render a compact final-eval packet for frozen-B1 residual candidates.

The residual rescue loop writes one ``closed_loop_report.json`` per matchup.
This script collects those reports, normalizes focal and reverse-direction
results into the residual policy's perspective, and writes thesis-friendly
CSV/JSON/Markdown plus PNG/PDF bar figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class EvalArtifactSpec:
    label: str
    artifact: str
    surface: str
    opponent: str
    direction: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_artifact(raw: str) -> EvalArtifactSpec:
    parts = str(raw).split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise argparse.ArgumentTypeError(
            "--artifact entries must be label=artifact_dir[:surface[:opponent[:direction]]]"
        )
    label = parts[0].strip()
    rest = parts[1].split(":")
    artifact = rest[0].strip()
    if not artifact:
        raise argparse.ArgumentTypeError("artifact directory must be non-empty")
    surface = rest[1].strip() if len(rest) > 1 and rest[1].strip() else ""
    opponent = rest[2].strip() if len(rest) > 2 and rest[2].strip() else ""
    direction = rest[3].strip() if len(rest) > 3 and rest[3].strip() else ""
    return EvalArtifactSpec(label=label, artifact=artifact, surface=surface, opponent=opponent, direction=direction)


def _pair_counts_from_residual_perspective(
    counts: Mapping[str, Any],
    *,
    residual_as_opponent: bool,
) -> dict[str, int]:
    normalized = {
        "2-0": int(counts.get("2-0", 0)),
        "1-1": int(counts.get("1-1", 0)),
        "0-2": int(counts.get("0-2", 0)),
        "mixed": int(counts.get("mixed", 0)),
    }
    if not residual_as_opponent:
        return normalized
    return {
        "2-0": normalized["0-2"],
        "1-1": normalized["1-1"],
        "0-2": normalized["2-0"],
        "mixed": normalized["mixed"],
    }


def _summarize_report(run_dir: Path, spec: EvalArtifactSpec) -> dict[str, Any]:
    report_path = run_dir / "eval" / spec.artifact / "closed_loop_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = _read_json(report_path)
    residual_as_opponent = bool(report.get("residual_as_opponent"))
    uncertainty = report.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ValueError(f"{report_path} is missing uncertainty")
    residual_mean = float(uncertainty.get("mean", 0.0))
    ci_low = float(uncertainty.get("ci_low", residual_mean))
    ci_high = float(uncertainty.get("ci_high", residual_mean))
    if residual_as_opponent:
        residual_mean = 1.0 - residual_mean
        ci_low, ci_high = 1.0 - ci_high, 1.0 - ci_low

    pair_summary = report.get("pair_class_summary")
    counts: Mapping[str, Any] = {}
    if isinstance(pair_summary, Mapping) and isinstance(pair_summary.get("pair_class_counts"), Mapping):
        counts = pair_summary["pair_class_counts"]
    residual_counts = _pair_counts_from_residual_perspective(counts, residual_as_opponent=residual_as_opponent)
    pair_count = int(pair_summary.get("pair_count", 0)) if isinstance(pair_summary, Mapping) else 0
    pair_score = (
        (2.0 * residual_counts["2-0"] + residual_counts["1-1"]) / max(2 * pair_count, 1)
        if pair_count
        else residual_mean
    )

    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    drift = report.get("residual_trace_drift") if isinstance(report.get("residual_trace_drift"), Mapping) else {}
    return {
        "label": spec.label,
        "artifact": spec.artifact,
        "surface": spec.surface,
        "opponent": spec.opponent or str(report.get("opponent_policy_id", "")),
        "direction": spec.direction or ("reverse" if residual_as_opponent else "focal"),
        "residual_as_opponent": residual_as_opponent,
        "pairs": int(report.get("pairs", pair_count)),
        "games": int(summary.get("games", 0)),
        "residual_win_rate": residual_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": max(residual_mean - ci_low, ci_high - residual_mean),
        "residual_pair_score": pair_score,
        "residual_2_0_pairs": residual_counts["2-0"],
        "residual_1_1_pairs": residual_counts["1-1"],
        "residual_0_2_pairs": residual_counts["0-2"],
        "residual_mixed_pairs": residual_counts["mixed"],
        "engine_errors": int(summary.get("engine_errors", 0)),
        "truncations": int(summary.get("truncations", 0)),
        "timeouts": int(summary.get("decision_limit_timeouts", 0))
        + int(summary.get("natural_timeouts", 0))
        + int(summary.get("tick_limit_timeouts", 0))
        + int(summary.get("no_progress_timeouts", 0)),
        "residual_family_drift_rate": float(drift.get("selected_family_differs_from_final_top1_rate", 0.0)),
        "trace_rows": int(report.get("runner_counters", {}).get("trace_rows", 0))
        if isinstance(report.get("runner_counters"), Mapping)
        else 0,
        "report_path": report_path.as_posix(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "surface",
        "opponent",
        "direction",
        "pairs",
        "games",
        "residual_win_rate",
        "ci_low",
        "ci_high",
        "residual_pair_score",
        "residual_2_0_pairs",
        "residual_1_1_pairs",
        "residual_0_2_pairs",
        "engine_errors",
        "truncations",
        "timeouts",
        "residual_family_drift_rate",
        "artifact",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# B1 Residual Final Eval Summary",
        "",
        "| Eval | Surface | Opponent | Dir | Pairs | Residual win | Pair classes | Drift |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        pair_classes = (
            f"{row['residual_2_0_pairs']}x 2-0, "
            f"{row['residual_1_1_pairs']}x 1-1, "
            f"{row['residual_0_2_pairs']}x 0-2"
        )
        lines.append(
            "| {label} | {surface} | {opponent} | {direction} | {pairs} | {win:.4f} | {pair_classes} | {drift:.4f} |".format(
                label=row["label"],
                surface=row["surface"],
                opponent=row["opponent"],
                direction=row["direction"],
                pairs=row["pairs"],
                win=float(row["residual_win_rate"]),
                pair_classes=pair_classes,
                drift=float(row["residual_family_drift_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Scores are normalized to the residual policy perspective.",
            "- Reverse-direction rows invert the B1-as-focal score so the table still reports residual win rate.",
            "- S3 is expected to be saturated by the public heuristic wrapper; S0 is a raw ablation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _surface_color(surface: str, opponent: str = "") -> str:
    opponent_key = opponent.lower()
    if "b3" in opponent_key or "b4" in opponent_key:
        return "#4b8f6a"
    key = surface.lower()
    if "s3" in key:
        return "#7a7f8a"
    if "s0" in key:
        return "#c47f2c"
    return "#3b6fb6"


def _render_figure(path_base: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    values = [float(row["residual_win_rate"]) for row in rows]
    lower = [max(0.0, float(row["residual_win_rate"]) - float(row["ci_low"])) for row in rows]
    upper = [max(0.0, float(row["ci_high"]) - float(row["residual_win_rate"])) for row in rows]
    colors = [_surface_color(str(row["surface"]), str(row["opponent"])) for row in rows]

    width = max(8.0, min(14.0, 1.25 * len(rows) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 5.2))
    positions = list(range(len(rows)))
    bars = ax.bar(positions, values, yerr=[lower, upper], color=colors, edgecolor="#243040", linewidth=0.8, capsize=4)
    ax.axhline(0.5, color="#222222", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Residual win rate")
    ax.set_title("Frozen-B1 residual final eval packet")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#d0d4da", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for bar, row in zip(bars, rows):
        height = bar.get_height()
        pair_text = f"{row['residual_2_0_pairs']}-{row['residual_1_1_pairs']}-{row['residual_0_2_pairs']}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.045, height + 0.035),
            f"{height:.2f}\n{pair_text}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(0.01, 0.505, "B1 parity", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8)
    fig.tight_layout()
    outputs: list[Path] = []
    for suffix in (".png", ".pdf"):
        out = path_base.with_suffix(suffix)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--artifact", action="append", type=_parse_artifact, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else run_dir / "figures" / "b1_residual_final_eval"
    rows = [_summarize_report(run_dir, spec) for spec in args.artifact]

    _write_csv(out_dir / "b1_residual_final_eval_summary.csv", rows)
    _write_json(
        out_dir / "b1_residual_final_eval_summary.json",
        {
            "format": "b1_residual_final_eval_summary_v1",
            "run_dir": run_dir.as_posix(),
            "rows": rows,
        },
    )
    _write_markdown(out_dir / "b1_residual_final_eval_summary.md", rows)
    figures = _render_figure(out_dir / "b1_residual_final_eval", rows)
    print(json.dumps({"out_dir": out_dir.as_posix(), "figure_outputs": [path.as_posix() for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
