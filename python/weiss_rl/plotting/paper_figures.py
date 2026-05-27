"""Paper figure generation from run artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.diagnostics.training_logger import TrainingLogger

__all__ = [
    "PAPER_FIGURE_IDS",
    "PAPER_FIGURE_STEMS",
    "render_paper_figures",
    "render_placeholder_figure",
    "render_public_demo_figures",
]

SUPPORTED_FORMATS = frozenset({"pdf", "png"})
PREFERRED_LEARNING_CURVE_FIELDS = (
    ("loss", "Loss"),
    ("value_loss", "Value loss"),
    ("actor_loss", "Actor loss"),
    ("entropy", "Entropy"),
    ("kl_divergence", "KL divergence"),
    ("throughput_samples_per_sec", "Samples/sec"),
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def render_placeholder_figure(out_path: Path) -> None:
    """Write a simple placeholder artifact until plotting is implemented."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("placeholder_figure\n", encoding="utf-8")


def render_public_demo_figures(*, final_eval_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Render clearly-labeled demo-only figure placeholders from public toy eval artifacts."""
    summary_path = final_eval_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing final_eval summary.json: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {summary_path}")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or not bool(metadata.get("demo_only", False)):
        raise ValueError("public-demo figure rendering requires final_eval metadata.demo_only=true")

    policy_ids_raw = payload.get("policy_ids", [])
    if not isinstance(policy_ids_raw, list):
        raise ValueError("final_eval summary policy_ids must be a list")
    policy_ids = [str(policy_id) for policy_id in policy_ids_raw]

    matrices = payload.get("matrices", {})
    if not isinstance(matrices, dict):
        raise ValueError("final_eval summary matrices must be an object")
    mean_matrix = matrices.get("mean", {})
    if not isinstance(mean_matrix, dict):
        raise ValueError("final_eval summary mean matrix must be an object")
    mean_values = mean_matrix.get("values", [])
    if not isinstance(mean_values, list):
        raise ValueError("final_eval summary mean.values must be a list")

    out_dir.mkdir(parents=True, exist_ok=True)

    placeholder_path = out_dir / "toy_demo_placeholder.txt"
    placeholder_path.write_text(
        "\n".join(
            (
                "toy_public_demo_placeholder_figure",
                str(metadata.get("warning", "demo-only artifact")),
                f"source_final_eval_dir={final_eval_dir.as_posix()}",
                f"policy_count={len(policy_ids)}",
                "",
            )
        ),
        encoding="utf-8",
    )

    matrix_csv_path = out_dir / "toy_demo_mean_matrix.csv"
    with matrix_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, mean_values, strict=True):
            if not isinstance(row, list):
                raise ValueError("final_eval summary mean.values rows must be lists")
            writer.writerow([focal_policy_id, *row])

    manifest_path = out_dir / "toy_demo_manifest.json"
    manifest_payload = {
        "kind": "toy_public_demo_figures_v1",
        "demo_only": True,
        "public_safe": True,
        "warning": metadata.get("warning", "demo-only artifact"),
        "source_final_eval_dir": final_eval_dir.as_posix(),
        "source_summary_path": summary_path.as_posix(),
        "policy_ids": policy_ids,
        "artifacts": {
            "placeholder": placeholder_path.name,
            "mean_matrix_csv": matrix_csv_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "placeholder": placeholder_path,
        "mean_matrix_csv": matrix_csv_path,
        "manifest": manifest_path,
    }


@dataclass(frozen=True)
class MatrixArtifact:
    labels: tuple[str, ...]
    values: FloatArray


@dataclass(frozen=True)
class SeatBiasArtifact:
    matchup_labels: tuple[str, ...]
    seat0_win_rates: FloatArray
    global_seat0_win_rate: float
    ci_low: float
    ci_high: float
    decisive_games: int


@dataclass(frozen=True)
class LearningCurveArtifact:
    updates: IntArray
    series: tuple[tuple[str, FloatArray], ...]


@dataclass(frozen=True)
class PaperFigureSpec:
    fig_id: str
    stem: str
    required_inputs: tuple[Path, ...]
    render: Callable[[Path], Figure]


def _render_matchup_heatmap(run_root: Path) -> Figure:
    layout = ArtifactLayout.from_run_dir(run_root)
    payoff_matrix = _load_square_matrix_csv(
        _resolve_existing_input(
            layout.final_eval_matrix_csv("mean"),
            layout.final_eval_payoff_matrix_csv("p_mean"),
        ),
        artifact_name="payoff matrix",
        minimum=0.0,
        maximum=1.0,
    )
    return _build_heatmap_figure(
        payoff_matrix,
        title="Final evaluation payoff matrix (p_mean)",
        colorbar_label="Mean score",
        cmap_name="coolwarm",
        vmin=0.0,
        vmax=1.0,
        value_format="{value:.2f}",
    )


def _render_truncation_heatmap(run_root: Path) -> Figure:
    layout = ArtifactLayout.from_run_dir(run_root)
    truncation_matrix = _load_square_matrix_csv(
        layout.truncation_heatmap_csv(),
        artifact_name="truncation heatmap",
        minimum=0.0,
        maximum=1.0,
    )
    return _build_heatmap_figure(
        truncation_matrix,
        title="Final evaluation truncation heatmap",
        colorbar_label="Truncation rate",
        cmap_name="magma",
        vmin=0.0,
        vmax=1.0,
        value_format="{value:.3f}",
    )


def _render_seat_bias(run_root: Path) -> Figure:
    layout = ArtifactLayout.from_run_dir(run_root)
    seat_bias = _load_seat_bias_json(layout.seat_bias_json())
    return _build_seat_bias_figure(seat_bias)


def _render_learning_curves(run_root: Path) -> Figure:
    layout = ArtifactLayout.from_run_dir(run_root)
    training_metrics_path = layout.training_logs_dir / "training_metrics.jsonl"
    if not training_metrics_path.is_file():
        interpolation_summary_path = _interpolation_summary_path(run_root)
        if interpolation_summary_path.is_file():
            return _render_interpolation_provenance(run_root, interpolation_summary_path)
    learning_curves = _load_learning_curves(training_metrics_path)
    return _build_learning_curves_figure(learning_curves)


def _paper_figure_specs() -> tuple[PaperFigureSpec, ...]:
    return (
        PaperFigureSpec(
            fig_id="matchup_heatmap",
            stem="fig_matchup_heatmap",
            required_inputs=(Path("eval/final_eval/payoff_matrices/p_mean.csv"),),
            render=_render_matchup_heatmap,
        ),
        PaperFigureSpec(
            fig_id="truncation_heatmap",
            stem="fig_truncation_heatmap",
            required_inputs=(Path("eval/diagnostics/truncation_heatmap_data.csv"),),
            render=_render_truncation_heatmap,
        ),
        PaperFigureSpec(
            fig_id="seat_bias",
            stem="fig_seat_bias",
            required_inputs=(Path("eval/diagnostics/seat_bias.json"),),
            render=_render_seat_bias,
        ),
        PaperFigureSpec(
            fig_id="learning_curves",
            stem="fig_learning_curves",
            required_inputs=(Path("training/logs/training_metrics.jsonl"),),
            render=_render_learning_curves,
        ),
    )


PAPER_FIGURE_IDS = tuple(spec.fig_id for spec in _paper_figure_specs())
PAPER_FIGURE_STEMS = tuple(spec.stem for spec in _paper_figure_specs())


def render_paper_figures(
    run_dir: Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    fig_id: str | None = None,
) -> tuple[Path, ...]:
    """Render paper figures for a run directory.

    When ``fig_id`` is supplied, only that stable figure ID is rendered.
    Otherwise all registered paper figures are rendered.
    """

    normalized_formats = _normalize_formats(formats)
    run_root = Path(run_dir)
    figure_specs = _resolve_figure_specs(fig_id)
    _validate_required_inputs(run_root, figure_specs)

    out_dir = run_root / "figures" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for spec in figure_specs:
        figure = spec.render(run_root)
        try:
            outputs.extend(_save_figure(figure, out_dir=out_dir, stem=spec.stem, formats=normalized_formats))
        finally:
            plt.close(figure)
    return tuple(outputs)


def _resolve_existing_input(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    candidate_list = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"missing required input artifact; checked: {candidate_list}")


def _resolve_figure_specs(fig_id: str | None) -> tuple[PaperFigureSpec, ...]:
    specs = _paper_figure_specs()
    if fig_id is None:
        return specs

    candidate = str(fig_id).strip().lower()
    if not candidate:
        raise ValueError("fig_id must be a non-empty string")

    for spec in specs:
        if spec.fig_id == candidate:
            return (spec,)

    allowed = ", ".join(PAPER_FIGURE_IDS)
    raise ValueError(f"unknown fig_id {fig_id!r}; expected one of: {allowed}")


def _validate_required_inputs(run_root: Path, figure_specs: Sequence[PaperFigureSpec]) -> None:
    missing: list[Path] = []
    seen: set[Path] = set()
    for spec in figure_specs:
        for relative_path in spec.required_inputs:
            artifact_path = run_root / relative_path
            if artifact_path in seen:
                continue
            seen.add(artifact_path)
            if not artifact_path.is_file():
                if spec.fig_id == "learning_curves" and _interpolation_summary_path(run_root).is_file():
                    continue
                missing.append(artifact_path)

    if not missing:
        return

    selected_ids = ", ".join(spec.fig_id for spec in figure_specs)
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(f"missing required input artifact(s) for fig-id selection {selected_ids}:\n{missing_lines}")


def _normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for fmt in formats:
        candidate = str(fmt).strip().lower()
        if not candidate:
            raise ValueError("figure format entries must be non-empty")
        if candidate not in SUPPORTED_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"unsupported figure format {fmt!r}; expected one of: {allowed}")
        if candidate not in normalized:
            normalized.append(candidate)
    if not normalized:
        raise ValueError("at least one output format is required")
    return tuple(normalized)


def _save_figure(figure: Figure, *, out_dir: Path, stem: str, formats: Sequence[str]) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    return written


def _build_heatmap_figure(
    matrix: MatrixArtifact,
    *,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    value_format: str,
) -> Figure:
    figure_size = max(6.0, 1.15 * len(matrix.labels))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix.values, cmap=cmap_name, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("Column policy")
    axis.set_ylabel("Row policy")
    axis.set_xticks(range(len(matrix.labels)), labels=matrix.labels, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.labels)), labels=matrix.labels)

    midpoint = vmin + ((vmax - vmin) / 2.0)
    for row_index in range(matrix.values.shape[0]):
        for col_index in range(matrix.values.shape[1]):
            value = float(matrix.values[row_index, col_index])
            text_color = "white" if value >= midpoint else "black"
            axis.text(
                col_index,
                row_index,
                value_format.format(value=value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    return figure


def _build_seat_bias_figure(seat_bias: SeatBiasArtifact) -> Figure:
    width = max(7.0, 1.3 * len(seat_bias.matchup_labels))
    figure, axis = plt.subplots(figsize=(width, 4.5))
    x_positions = np.arange(len(seat_bias.matchup_labels), dtype=np.float64)
    colors = ["tab:blue" if rate >= 0.5 else "tab:orange" for rate in seat_bias.seat0_win_rates]

    axis.bar(x_positions, seat_bias.seat0_win_rates, color=colors)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="No seat bias")
    axis.axhline(
        seat_bias.global_seat0_win_rate,
        color="tab:green",
        linestyle="-",
        linewidth=1.2,
        label="Global seat0 win rate",
    )
    axis.axhspan(seat_bias.ci_low, seat_bias.ci_high, color="tab:green", alpha=0.12, label="Global CI")
    axis.set_title(
        "Seat bias by matchup\n"
        f"global seat0 win rate={seat_bias.global_seat0_win_rate:.3f}, decisive games={seat_bias.decisive_games}"
    )
    axis.set_ylabel("Seat0 win rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x_positions, labels=seat_bias.matchup_labels, rotation=30, ha="right")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _build_learning_curves_figure(artifact: LearningCurveArtifact) -> Figure:
    subplot_count = len(artifact.series)
    columns = 2 if subplot_count > 1 else 1
    rows = int(math.ceil(subplot_count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.5 * columns, 3.6 * rows), squeeze=False)
    flat_axes = list(axes.flat)

    for axis, (label, values) in zip(flat_axes, artifact.series, strict=True):
        axis.plot(artifact.updates, values, linewidth=2.0)
        axis.set_title(label)
        axis.set_xlabel("Update")
        axis.grid(alpha=0.25)

    for axis in flat_axes[subplot_count:]:
        axis.set_visible(False)

    figure.suptitle("Training learning curves", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def _interpolation_summary_path(run_root: Path) -> Path:
    return Path(run_root) / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json"


def _render_interpolation_provenance(run_root: Path, summary_path: Path) -> Figure:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint interpolation summary must be a JSON object: {summary_path}")
    second_weight = _parse_bounded_float(
        payload.get("second_weight"),
        path=summary_path,
        field_name="second_weight",
        minimum=0.0,
        maximum=1.0,
    )
    first_weight = 1.0 - second_weight
    first_checkpoint = _require_nonempty_text(
        payload.get("first_checkpoint"),
        path=summary_path,
        field_name="first_checkpoint",
    )
    second_checkpoint = _require_nonempty_text(
        payload.get("second_checkpoint"),
        path=summary_path,
        field_name="second_checkpoint",
    )

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    labels = ("First checkpoint", "Second checkpoint")
    weights = (first_weight, second_weight)
    bars = axis.barh(labels, weights, color=("tab:blue", "tab:orange"))
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Interpolation weight")
    axis.set_title("Selected checkpoint interpolation provenance")
    axis.grid(axis="x", alpha=0.25)
    for bar, weight in zip(bars, weights, strict=True):
        axis.text(
            min(float(weight) + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2.0,
            f"{weight:.2f}",
            va="center",
            ha="left" if weight < 0.96 else "right",
        )

    summary_lines = [
        f"Run: {Path(run_root).name}",
        f"First: {_short_artifact_path(first_checkpoint)}",
        f"Second: {_short_artifact_path(second_checkpoint)}",
        "This run is an interpolated selected checkpoint; no standalone training curve exists.",
    ]
    figure.text(0.02, 0.02, "\n".join(summary_lines), ha="left", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    return figure


def _short_artifact_path(path_text: str, *, max_parts: int = 4) -> str:
    parts = Path(path_text).parts
    if len(parts) <= max_parts:
        return path_text
    return ".../" + "/".join(parts[-max_parts:])


def _load_square_matrix_csv(path: Path, *, artifact_name: str, minimum: float, maximum: float) -> MatrixArtifact:
    _require_existing_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"{artifact_name} CSV is empty: {path}")

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"{artifact_name} CSV must contain a header row and at least one labeled column: {path}")

    labels = tuple(cell.strip() for cell in header[1:])
    if any(not label for label in labels):
        raise ValueError(f"{artifact_name} CSV contains an empty column label: {path}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{artifact_name} CSV contains duplicate column labels: {path}")
    if len(rows) != len(labels) + 1:
        raise ValueError(
            f"{artifact_name} CSV must contain exactly {len(labels)} data rows to match the labeled columns: {path}"
        )

    values = np.zeros((len(labels), len(labels)), dtype=np.float64)
    row_labels: list[str] = []
    expected_width = len(header)
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_width:
            raise ValueError(
                f"{artifact_name} CSV row {row_number} has {len(row)} columns; expected {expected_width}: {path}"
            )
        row_label = row[0].strip()
        if not row_label:
            raise ValueError(f"{artifact_name} CSV row {row_number} is missing its row label: {path}")
        row_labels.append(row_label)

        for value_index, raw_value in enumerate(row[1:]):
            values[row_number - 2, value_index] = _parse_bounded_float_token(
                raw_value,
                path=path,
                field_name=f"row {row_number} column {value_index + 2}",
                minimum=minimum,
                maximum=maximum,
            )

    if tuple(row_labels) != labels:
        raise ValueError(f"{artifact_name} CSV row labels must match the column labels in the same order: {path}")
    return MatrixArtifact(labels=labels, values=values)


def _load_seat_bias_json(path: Path) -> SeatBiasArtifact:
    _require_existing_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"seat bias payload must be a JSON object: {path}")

    global_section = _require_mapping(payload, "global", path=path)
    matchups_payload = payload.get("matchups")
    if not isinstance(matchups_payload, list) or not matchups_payload:
        raise ValueError(f"seat bias payload must include a non-empty 'matchups' list: {path}")

    global_seat0_win_rate = _parse_bounded_float(
        global_section.get("seat0_win_rate"),
        path=path,
        field_name="global.seat0_win_rate",
        minimum=0.0,
        maximum=1.0,
    )
    ci_low = _parse_bounded_float(
        global_section.get("ci_low"),
        path=path,
        field_name="global.ci_low",
        minimum=0.0,
        maximum=1.0,
    )
    ci_high = _parse_bounded_float(
        global_section.get("ci_high"),
        path=path,
        field_name="global.ci_high",
        minimum=0.0,
        maximum=1.0,
    )
    decisive_games = _parse_nonnegative_int(
        global_section.get("decisive_games"),
        path=path,
        field_name="global.decisive_games",
    )

    if not (ci_low <= global_seat0_win_rate <= ci_high):
        raise ValueError(f"seat bias global interval must bracket the global seat0 win rate: {path}")

    matchup_labels: list[str] = []
    seat0_win_rates: list[float] = []
    for index, matchup_payload in enumerate(matchups_payload, start=1):
        if not isinstance(matchup_payload, dict):
            raise ValueError(f"seat bias matchup entry {index} must be an object: {path}")
        policy_a = _require_nonempty_text(
            matchup_payload.get("policy_a"),
            path=path,
            field_name=f"matchups[{index}].policy_a",
        )
        policy_b = _require_nonempty_text(
            matchup_payload.get("policy_b"),
            path=path,
            field_name=f"matchups[{index}].policy_b",
        )
        seat0_rate = _parse_bounded_float(
            matchup_payload.get("seat0_win_rate"),
            path=path,
            field_name=f"matchups[{index}].seat0_win_rate",
            minimum=0.0,
            maximum=1.0,
        )
        seat1_rate = _parse_bounded_float(
            matchup_payload.get("seat1_win_rate"),
            path=path,
            field_name=f"matchups[{index}].seat1_win_rate",
            minimum=0.0,
            maximum=1.0,
        )
        _parse_nonnegative_int(
            matchup_payload.get("decisive_games"),
            path=path,
            field_name=f"matchups[{index}].decisive_games",
        )
        if not math.isclose(seat0_rate + seat1_rate, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"seat bias matchup entry {index} must satisfy seat0_win_rate + seat1_win_rate == 1: {path}"
            )
        matchup_labels.append(f"{policy_a} vs {policy_b}")
        seat0_win_rates.append(seat0_rate)

    if len(set(matchup_labels)) != len(matchup_labels):
        raise ValueError(f"seat bias matchup labels must be unique: {path}")

    return SeatBiasArtifact(
        matchup_labels=tuple(matchup_labels),
        seat0_win_rates=np.asarray(seat0_win_rates, dtype=np.float64),
        global_seat0_win_rate=global_seat0_win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        decisive_games=decisive_games,
    )


def _load_learning_curves(path: Path) -> LearningCurveArtifact:
    _require_existing_file(path)
    records = TrainingLogger.read_jsonl(path)
    if not records:
        raise ValueError(f"training metrics JSONL is empty: {path}")

    updates = np.asarray(
        [
            _parse_nonnegative_int(
                record.get("update_count"),
                path=path,
                field_name=f"record[{index}].update_count",
            )
            for index, record in enumerate(records, start=1)
        ],
        dtype=np.int64,
    )
    if not np.all(np.diff(updates) > 0):
        raise ValueError(f"training metrics update_count values must be strictly increasing: {path}")

    plottable_series: list[tuple[str, FloatArray]] = []
    for field_name, label in PREFERRED_LEARNING_CURVE_FIELDS:
        if not any(field_name in record for record in records):
            continue
        values: list[float] = []
        for index, record in enumerate(records, start=1):
            if field_name not in record:
                raise ValueError(f"training metrics field {field_name!r} is missing from record {index}: {path}")
            values.append(
                _parse_finite_number(record[field_name], path=path, field_name=f"record[{index}].{field_name}")
            )
        plottable_series.append((label, np.asarray(values, dtype=np.float64)))
        if len(plottable_series) == 4:
            break

    if not plottable_series:
        raise ValueError(f"training metrics JSONL does not contain any supported learning-curve metrics: {path}")

    return LearningCurveArtifact(updates=updates, series=tuple(plottable_series))


def _require_existing_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")


def _require_mapping(payload: dict[str, Any], key: str, *, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"seat bias payload field {key!r} must be an object: {path}")
    return value


def _require_nonempty_text(value: Any, *, path: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string: {path}")
    return value.strip()


def _parse_nonnegative_int(value: Any, *, path: Path, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer: {path}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative: {path}")
    return value


def _parse_finite_number(value: Any, *, path: Path, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric: {path}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite: {path}")
    return result


def _parse_bounded_float_token(
    value: str,
    *,
    path: Path,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    token = value.strip()
    if not token:
        raise ValueError(f"{field_name} must be non-empty: {path}")
    try:
        parsed = float(token)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric: {path}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite: {path}")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}]: {path}")
    return parsed


def _parse_bounded_float(
    value: Any,
    *,
    path: Path,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    parsed = _parse_finite_number(value, path=path, field_name=field_name)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}]: {path}")
    return parsed
