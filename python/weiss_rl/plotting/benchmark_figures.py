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
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
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
    final_eval_degraded: bool


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
                        "final_eval_degraded": record.final_eval_degraded,
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
                "final_eval_degraded",
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
                    str(record.final_eval_degraded).lower(),
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
    final_score_vs_b0 = _extract_focal_score_vs_b0(summary_path) if summary_path.is_file() else None
    final_eval_degraded = _extract_final_eval_degraded(summary_path) if summary_path.is_file() else False
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
        final_eval_degraded=final_eval_degraded,
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
    axis.set_title("Final focal score vs B0 RandomLegal")
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


def _extract_focal_score_vs_b0(summary_path: Path) -> float | None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_ids = list(payload.get("policy_ids", ()))
    if "B0 RandomLegal" not in policy_ids:
        return None
    focal_policy_id = _resolve_focal_policy_id(payload, policy_ids=policy_ids)
    if focal_policy_id is None:
        return None
    mean_payload = payload.get("matrices", {}).get("mean", {})
    values = mean_payload.get("values", ())
    b0_index = policy_ids.index("B0 RandomLegal")
    focal_index = policy_ids.index(focal_policy_id)
    return float(values[focal_index][b0_index])


def _extract_final_eval_degraded(summary_path: Path) -> bool:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    if bool(metadata.get("degraded", False)):
        return True
    selection = metadata.get("selection", {})
    if not isinstance(selection, dict):
        return False
    return bool(selection.get("degraded", False))


def _resolve_focal_policy_id(payload: dict[str, Any], *, policy_ids: list[str]) -> str | None:
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("focal_policy_id", "recommended_focal_policy_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value in policy_ids:
                return value
        selection = metadata.get("selection", {})
        if isinstance(selection, dict):
            value = selection.get("focal_policy_id")
            if isinstance(value, str) and value in policy_ids:
                return value
    eligible_policy_ids = [str(policy_id) for policy_id in policy_ids if str(policy_id) not in _BASELINE_POLICY_IDS]
    if len(eligible_policy_ids) == 1:
        return eligible_policy_ids[0]
    return None


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
    league_config = dict(config_root.get("league", {}))
    environment_config = dict(config_root.get("environment", {}))
    algorithm = str(training_config.get("algorithm", "unknown"))
    experiment_role = str(experiment_config.get("role", "main"))
    recurrent_core = str(model_config.get("recurrent_core", "gru"))
    encoder_kind = str(model_config.get("encoder_kind", "mlp"))

    if experiment_role == "ablation_teacher_fade" or _looks_like_teacher_fade_ablation(
        model_config=model_config,
        training_config=training_config,
    ):
        return "thesis_ablation_teacher_fade", "Thesis ablation: teacher fade"
    if experiment_role == "ablation_no_tactical_bias" or _looks_like_no_tactical_bias_ablation(model_config=model_config):
        return "thesis_ablation_no_tactical_bias", "Thesis ablation: no tactical bias"
    if experiment_role == "ablation_no_b1_cutoff" or _looks_like_no_b1_cutoff_ablation(league_config=league_config):
        return "thesis_ablation_no_b1_cutoff", "Thesis ablation: no B1 cutoff"
    if experiment_role == "thesis_multideck" or _looks_like_multideck(environment_config=environment_config):
        return "thesis_multideck", "Thesis model multideck"
    if experiment_role == "baseline_noleague_multideck":
        return "thesis_multideck_no_league", "Thesis model multideck no league"
    if experiment_role == "baseline_noleague_ablation_teacher_fade":
        return "thesis_ablation_teacher_fade_no_league", "Thesis ablation: teacher fade no league"
    if experiment_role == "baseline_noleague_ablation_no_tactical_bias":
        return "thesis_ablation_no_tactical_bias_no_league", "Thesis ablation: no tactical bias no league"
    if algorithm == "ppo_lite_masked_v1":
        return "ppo_lite", "PPO-lite"
    if experiment_role == "baseline_noleague":
        return "impala_no_league", "IMPALA no league"
    if recurrent_core == "none":
        return "impala_no_recurrence", "IMPALA no recurrence"
    if encoder_kind == "typed_v1":
        return "impala_typed", "IMPALA typed"
    if _looks_like_thesis_model(model_config=model_config, training_config=training_config):
        return "thesis_model", "Thesis model"
    return "impala_main", "IMPALA main"


def _looks_like_thesis_model(*, model_config: dict[str, Any], training_config: dict[str, Any]) -> bool:
    return (
        str(model_config.get("encoder_kind", "mlp")) == "structured_v2"
        or str(training_config.get("actor_policy_backend", "model")) == "heuristic_public"
    )


def _looks_like_teacher_fade_ablation(*, model_config: dict[str, Any], training_config: dict[str, Any]) -> bool:
    structured_aux = training_config.get("structured_aux", {})
    if not isinstance(structured_aux, dict):
        structured_aux = {}
    scheduled_values = (
        model_config.get("public_heuristic_logit_bias_end_updates", -1),
        training_config.get("actor_heuristic_end_updates", -1),
        structured_aux.get("teacher_public_heuristic_end_updates", -1),
    )
    return any(isinstance(value, int) and value >= 0 for value in scheduled_values)


def _looks_like_no_tactical_bias_ablation(*, model_config: dict[str, Any]) -> bool:
    families = model_config.get("public_heuristic_logit_bias_families", ())
    return (
        isinstance(families, list)
        and not families
        and float(model_config.get("public_heuristic_logit_bias_scale", 0.0)) == 1.0
        and float(model_config.get("public_heuristic_logit_bias_final_scale", 0.0)) == 1.0
    )


def _looks_like_no_b1_cutoff_ablation(*, league_config: dict[str, Any]) -> bool:
    sampling_config = league_config.get("sampling", {})
    if not isinstance(sampling_config, dict):
        return False
    return int(sampling_config.get("noleague_baseline_mix_end_updates", 0)) < 0


def _looks_like_multideck(*, environment_config: dict[str, Any]) -> bool:
    return bool(environment_config.get("deck_pool")) or bool(environment_config.get("opponent_deck_pool"))


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
