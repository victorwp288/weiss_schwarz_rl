"""Cross-run baseline and scaling comparison artifacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

_BASELINE_POLICY_IDS = frozenset(
    {
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "b0_randomlegal",
        "b1_noleague_baseline",
    }
)


@dataclass(frozen=True, slots=True)
class RunBenchmarkRecord:
    run_dir: Path
    label: str
    method_id: str
    method_label: str
    algorithm: str
    recurrent_core: str
    training_mode: str
    encoder_kind: str
    update_counts: np.ndarray
    loss_values: np.ndarray
    throughput_time: np.ndarray
    throughput_values: np.ndarray
    final_score_vs_b0: float | None


def render_benchmark_figures(
    *,
    run_dirs: list[Path],
    out_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> tuple[Path, ...]:
    records = [_load_run_benchmark_record(Path(run_dir)) for run_dir in run_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(_write_learning_curve_figure(records, out_dir=out_dir, formats=formats))
    written.extend(_write_throughput_figure(records, out_dir=out_dir, formats=formats))
    written.extend(_write_final_score_figure(records, out_dir=out_dir, formats=formats))
    written.extend(_write_summary_table(records, out_dir=out_dir))
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "kind": "run_benchmark_summary_v1",
                "methods": _method_summary_payload(records),
                "runs": [
                    {
                        "label": record.label,
                        "run_dir": record.run_dir.as_posix(),
                        "method_id": record.method_id,
                        "method_label": record.method_label,
                        "algorithm": record.algorithm,
                        "recurrent_core": record.recurrent_core,
                        "training_mode": record.training_mode,
                        "encoder_kind": record.encoder_kind,
                        "final_score_vs_b0": record.final_score_vs_b0,
                    }
                    for record in records
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(summary_path)
    runs_csv_path = out_dir / "benchmark_runs.csv"
    with runs_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "method_id",
                "method_label",
                "algorithm",
                "recurrent_core",
                "training_mode",
                "encoder_kind",
                "final_score_vs_b0",
                "run_dir",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.label,
                    record.method_id,
                    record.method_label,
                    record.algorithm,
                    record.recurrent_core,
                    record.training_mode,
                    record.encoder_kind,
                    "" if record.final_score_vs_b0 is None else f"{record.final_score_vs_b0:.6f}",
                    record.run_dir.as_posix(),
                ]
            )
    written.append(runs_csv_path)
    return tuple(written)


def _load_run_benchmark_record(run_dir: Path) -> RunBenchmarkRecord:
    training_metrics_path = run_dir / "training" / "logs" / "training_metrics.jsonl"
    training_records = _read_jsonl(training_metrics_path)
    update_counts = np.asarray([int(record["update_count"]) for record in training_records], dtype=np.int64)
    loss_values = np.asarray([float(record.get("loss", 0.0)) for record in training_records], dtype=np.float64)

    performance_path = run_dir / "training" / "logs" / "performance.jsonl"
    if performance_path.is_file():
        performance_records = _read_jsonl(performance_path)
        throughput_time = np.asarray(
            [float(record.get("wall_clock_seconds", 0.0)) for record in performance_records],
            dtype=np.float64,
        )
        throughput_values = np.asarray(
            [float(record.get("actor_env_steps_per_sec", 0.0)) for record in performance_records],
            dtype=np.float64,
        )
    else:
        throughput_time = np.asarray(
            [float(record.get("wall_clock_seconds", 0.0)) for record in training_records],
            dtype=np.float64,
        )
        throughput_values = np.asarray(
            [float(record.get("throughput_samples_per_sec", 0.0)) for record in training_records],
            dtype=np.float64,
        )

    config_payload = json.loads((run_dir / "config_canonical.json").read_text(encoding="utf-8"))
    config_root = config_payload.get("config", {})
    model_config = config_root.get("model", {})
    training_config = config_root.get("training", {})
    experiment_config = config_root.get("experiment", {})
    summary_path = run_dir / "eval" / "final_eval" / "summary.json"
    final_score_vs_b0 = _extract_best_score_vs_b0(summary_path) if summary_path.is_file() else None
    method_id, method_label = _method_from_config(config_root)
    return RunBenchmarkRecord(
        run_dir=run_dir,
        label=run_dir.name,
        method_id=method_id,
        method_label=method_label,
        algorithm=str(training_config.get("algorithm", "unknown")),
        recurrent_core=str(model_config.get("recurrent_core", "gru")),
        training_mode=str(experiment_config.get("role", "main")),
        encoder_kind=str(model_config.get("encoder_kind", "mlp")),
        update_counts=update_counts,
        loss_values=loss_values,
        throughput_time=throughput_time,
        throughput_values=throughput_values,
        final_score_vs_b0=final_score_vs_b0,
    )


def _write_learning_curve_figure(
    records: list[RunBenchmarkRecord],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for method in _group_records(records):
        grid, mean, low, high = _aggregate_curve(method.records, x_attr="update_counts", y_attr="loss_values")
        axis.plot(grid, mean, linewidth=2.2, label=f"{method.method_label} (n={len(method.records)})")
        if low is not None and high is not None:
            axis.fill_between(grid, low, high, alpha=0.15)
    axis.set_title("Seed-aggregated loss vs update")
    axis.set_xlabel("Update")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return _save_figure(figure, out_dir=out_dir, stem="fig_benchmark_loss", formats=formats)


def _write_throughput_figure(
    records: list[RunBenchmarkRecord],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for method in _group_records(records):
        grid, mean, low, high = _aggregate_curve(method.records, x_attr="throughput_time", y_attr="throughput_values")
        axis.plot(grid, mean, linewidth=2.2, label=f"{method.method_label} (n={len(method.records)})")
        if low is not None and high is not None:
            axis.fill_between(grid, low, high, alpha=0.15)
    axis.set_title("Seed-aggregated runtime throughput vs wall clock")
    axis.set_xlabel("Wall clock seconds")
    axis.set_ylabel("Env steps/sec")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return _save_figure(figure, out_dir=out_dir, stem="fig_benchmark_throughput", formats=formats)


def _write_final_score_figure(
    records: list[RunBenchmarkRecord],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    methods = _group_records(records)
    labels = [method.method_label for method in methods]
    values = [method.final_score_mean for method in methods]
    errors = [method.final_score_std for method in methods]
    bars = axis.bar(np.arange(len(methods), dtype=np.float64), values, yerr=errors, color="tab:blue", capsize=4)
    axis.set_title("Best non-baseline final score vs B0 RandomLegal")
    axis.set_ylabel("Final-eval mean score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(len(methods), dtype=np.float64), labels=labels, rotation=25, ha="right")
    for bar, method in zip(bars, methods, strict=True):
        if math.isnan(method.final_score_mean):
            axis.text(bar.get_x() + (bar.get_width() / 2.0), 0.02, "n/a", ha="center", va="bottom", fontsize=8)
        else:
            axis.text(
                bar.get_x() + (bar.get_width() / 2.0),
                min(0.98, method.final_score_mean + method.final_score_std + 0.02),
                f"n={len(method.records)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, out_dir=out_dir, stem="fig_benchmark_final_score", formats=formats)


def _write_summary_table(records: list[RunBenchmarkRecord], *, out_dir: Path) -> list[Path]:
    methods = _group_records(records)
    csv_path = out_dir / "benchmark_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method_id",
                "method_label",
                "seed_count",
                "algorithm",
                "recurrent_core",
                "training_mode",
                "encoder_kind",
                "final_score_vs_b0_mean",
                "final_score_vs_b0_std",
                "final_loss_mean",
                "peak_throughput_mean",
            ]
        )
        for method in methods:
            writer.writerow(
                [
                    method.method_id,
                    method.method_label,
                    len(method.records),
                    method.algorithm,
                    method.recurrent_core,
                    method.training_mode,
                    method.encoder_kind,
                    "" if math.isnan(method.final_score_mean) else f"{method.final_score_mean:.6f}",
                    "" if math.isnan(method.final_score_std) else f"{method.final_score_std:.6f}",
                    f"{method.final_loss_mean:.6f}",
                    f"{method.peak_throughput_mean:.6f}",
                ]
            )
    markdown_path = out_dir / "benchmark_summary.md"
    markdown_lines = [
        "| Method | Seeds | Final score vs B0 | Std | Final loss | Peak throughput |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        score_text = "n/a" if math.isnan(method.final_score_mean) else f"{method.final_score_mean:.3f}"
        std_text = "n/a" if math.isnan(method.final_score_std) else f"{method.final_score_std:.3f}"
        markdown_lines.append(
            f"| {method.method_label} | {len(method.records)} | {score_text} | {std_text} | "
            f"{method.final_loss_mean:.3f} | {method.peak_throughput_mean:.1f} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return [csv_path, markdown_path]


def _extract_best_score_vs_b0(summary_path: Path) -> float | None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_ids = list(payload.get("policy_ids", ()))
    if "B0 RandomLegal" not in policy_ids:
        return None
    mean_payload = payload.get("matrices", {}).get("mean", {})
    values = mean_payload.get("values", ())
    b0_index = policy_ids.index("B0 RandomLegal")
    best_score: float | None = None
    for policy_index, policy_id in enumerate(policy_ids):
        if str(policy_id) in _BASELINE_POLICY_IDS:
            continue
        candidate = float(values[policy_index][b0_index])
        if best_score is None or candidate > best_score:
            best_score = candidate
    return best_score


@dataclass(frozen=True, slots=True)
class _MethodAggregate:
    method_id: str
    method_label: str
    algorithm: str
    recurrent_core: str
    training_mode: str
    encoder_kind: str
    records: tuple[RunBenchmarkRecord, ...]
    final_score_mean: float
    final_score_std: float
    final_loss_mean: float
    peak_throughput_mean: float


def _group_records(records: list[RunBenchmarkRecord]) -> list[_MethodAggregate]:
    grouped: dict[str, list[RunBenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.method_id, []).append(record)
    aggregates: list[_MethodAggregate] = []
    for method_id, method_records in grouped.items():
        ordered = tuple(sorted(method_records, key=lambda record: record.label))
        final_scores = np.asarray(
            [record.final_score_vs_b0 for record in ordered if record.final_score_vs_b0 is not None],
            dtype=np.float64,
        )
        final_losses = np.asarray([record.loss_values[-1] for record in ordered], dtype=np.float64)
        peak_throughputs = np.asarray([float(np.max(record.throughput_values)) for record in ordered], dtype=np.float64)
        first = ordered[0]
        aggregates.append(
            _MethodAggregate(
                method_id=method_id,
                method_label=first.method_label,
                algorithm=first.algorithm,
                recurrent_core=first.recurrent_core,
                training_mode=first.training_mode,
                encoder_kind=first.encoder_kind,
                records=ordered,
                final_score_mean=(float(np.mean(final_scores)) if final_scores.size else float("nan")),
                final_score_std=(float(np.std(final_scores)) if final_scores.size else float("nan")),
                final_loss_mean=float(np.mean(final_losses)),
                peak_throughput_mean=float(np.mean(peak_throughputs)),
            )
        )
    return sorted(aggregates, key=lambda method: method.method_label)


def _aggregate_curve(
    records: tuple[RunBenchmarkRecord, ...],
    *,
    x_attr: str,
    y_attr: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if len(records) == 1:
        x = np.asarray(getattr(records[0], x_attr), dtype=np.float64)
        y = np.asarray(getattr(records[0], y_attr), dtype=np.float64)
        return x, y, None, None

    min_x = max(float(np.min(np.asarray(getattr(record, x_attr), dtype=np.float64))) for record in records)
    max_x = min(float(np.max(np.asarray(getattr(record, x_attr), dtype=np.float64))) for record in records)
    if not min_x < max_x:
        grid = np.asarray(getattr(records[0], x_attr), dtype=np.float64)
        stacked = np.stack([np.asarray(getattr(record, y_attr), dtype=np.float64) for record in records], axis=0)
        return grid, stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)

    grid = np.linspace(min_x, max_x, num=128, dtype=np.float64)
    interpolated = np.stack(
        [
            np.interp(
                grid,
                np.asarray(getattr(record, x_attr), dtype=np.float64),
                np.asarray(getattr(record, y_attr), dtype=np.float64),
            )
            for record in records
        ],
        axis=0,
    )
    return grid, interpolated.mean(axis=0), interpolated.min(axis=0), interpolated.max(axis=0)


def _method_from_config(config_root: dict[str, Any]) -> tuple[str, str]:
    model_config = dict(config_root.get("model", {}))
    training_config = dict(config_root.get("training", {}))
    experiment_config = dict(config_root.get("experiment", {}))
    algorithm = str(training_config.get("algorithm", "unknown"))
    experiment_role = str(experiment_config.get("role", "main"))
    recurrent_core = str(model_config.get("recurrent_core", "gru"))
    encoder_kind = str(model_config.get("encoder_kind", "mlp"))

    if algorithm == "ppo_lite_masked_v1":
        return "ppo_lite", "PPO-lite"
    if experiment_role == "baseline_noleague":
        return "impala_no_league", "IMPALA no league"
    if recurrent_core == "none":
        return "impala_no_recurrence", "IMPALA no recurrence"
    if encoder_kind == "typed_v1":
        return "impala_typed", "IMPALA typed"
    return "impala_main", "IMPALA main"


def _method_summary_payload(records: list[RunBenchmarkRecord]) -> list[dict[str, Any]]:
    return [
        {
            "method_id": method.method_id,
            "method_label": method.method_label,
            "seed_count": len(method.records),
            "algorithm": method.algorithm,
            "recurrent_core": method.recurrent_core,
            "training_mode": method.training_mode,
            "encoder_kind": method.encoder_kind,
            "final_score_vs_b0_mean": None if math.isnan(method.final_score_mean) else method.final_score_mean,
            "final_score_vs_b0_std": None if math.isnan(method.final_score_std) else method.final_score_std,
            "final_loss_mean": method.final_loss_mean,
            "peak_throughput_mean": method.peak_throughput_mean,
        }
        for method in _group_records(records)
    ]


def _save_figure(figure: Any, *, out_dir: Path, stem: str, formats: tuple[str, ...]) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object records in {path}")
            records.append(payload)
    if not records:
        raise ValueError(f"expected at least one JSONL record in {path}")
    return records
