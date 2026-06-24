"""Run-level TensorBoard helpers for training and evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter

_SummaryWriterClass: Any

try:
    from torch.utils.tensorboard import SummaryWriter as _ImportedSummaryWriter
except Exception as exc:  # pragma: no cover - exercised via disabled logger fallback
    _SummaryWriterClass = None
    _SUMMARY_WRITER_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - trivial branch
    _SummaryWriterClass = _ImportedSummaryWriter
    _SUMMARY_WRITER_IMPORT_ERROR = None


def _json_markdown(payload: Any) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True)
    return f"```json\n{text}\n```"


def _tag_part(value: Any) -> str:
    text = str(value).strip()
    sanitized = []
    for character in text:
        if character.isalnum() or character in {"_", "-"}:
            sanitized.append(character)
        else:
            sanitized.append("_")
    return "".join(sanitized) or "unknown"


def _as_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scalar = float(value)
        if math.isfinite(scalar):
            return scalar
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            scalar = float(stripped)
        except ValueError:
            return None
        if math.isfinite(scalar):
            return scalar
    return None


def _iter_numeric_mapping(payload: Mapping[str, Any], *, prefix: str) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key in sorted(payload):
        tag = f"{prefix}/{_tag_part(key)}" if prefix else _tag_part(key)
        value = payload[key]
        scalar = _as_scalar(value)
        if scalar is not None:
            items.append((tag, scalar))
            continue
        if isinstance(value, Mapping):
            items.extend(_iter_numeric_mapping(value, prefix=tag))
    return items


def _matrix_array(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2:
        return None
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        return None
    return array


def _policy_metric_rows(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        value_field = next(
            (
                field
                for field in reader.fieldnames
                if field not in {"policy_id", "policy", "display_name"} and field is not None
            ),
            None,
        )
        if value_field is None:
            return rows
        for row in reader:
            policy_id = str(row.get("policy_id", row.get("policy", ""))).strip()
            scalar = _as_scalar(row.get(value_field))
            if policy_id and scalar is not None:
                rows.append((policy_id, scalar))
    return rows


class TensorBoardLogger:
    """Structured TensorBoard logger for run-level metrics and summaries."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = _SummaryWriterClass is not None
        self._writer: _TorchSummaryWriter | None = (
            None if _SummaryWriterClass is None else _SummaryWriterClass(log_dir=str(self.log_dir), flush_secs=10)
        )
        self._logged_text_tags: set[str] = set()
        self._logged_checkpoint_tracker_payloads: set[str] = set()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def log_run_context(
        self,
        *,
        manifest: Mapping[str, Any],
        environment: Mapping[str, Any],
        run_summary: Mapping[str, Any],
        determinism_report: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self.log_text("run/manifest", manifest)
        self.log_text("run/environment", environment)
        self.log_text("run/run_summary", run_summary)
        self.log_text("run/determinism_report", determinism_report)
        self.log_text("run/config_canonical", manifest.get("config_canonical", {}))
        self.log_text("run/simulator", manifest.get("simulator", {}))
        self._log_scalar(
            "run/policy_set_selection_count", float(len(cast(list[Any], manifest.get("policy_set_selection", [])))), 0
        )
        for tag, value in _iter_numeric_mapping(
            cast(Mapping[str, Any], manifest.get("hardware", {})), prefix="run/hardware"
        ):
            self._log_scalar(tag, value, 0)
        for tag, value in _iter_numeric_mapping(
            cast(Mapping[str, Any], manifest.get("evaluation_pinning", {})),
            prefix="run/evaluation_pinning",
        ):
            self._log_scalar(tag, value, 0)

    def log_training_step(
        self,
        *,
        update_count: int,
        policy_version: int,
        wall_clock_seconds: float,
        metrics: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self._log_scalar(
            "run/policy_version", float(policy_version), update_count, wall_clock_seconds=wall_clock_seconds
        )
        self._log_scalar(
            "run/wall_clock_seconds", float(wall_clock_seconds), update_count, wall_clock_seconds=wall_clock_seconds
        )
        for key, value in sorted(metrics.items()):
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            self._log_scalar(
                self._training_tag_for_key(key),
                scalar,
                update_count,
                wall_clock_seconds=wall_clock_seconds,
            )

    def log_checkpoint_tracker(self, tracker: Mapping[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        dedupe_key = f"{int(step)}:{json.dumps(tracker, sort_keys=True, default=str)}"
        if dedupe_key in self._logged_checkpoint_tracker_payloads:
            return
        self.log_text("checkpoint/tracker", tracker, step=step)
        for alias in ("latest", "best", "observed_best"):
            record = tracker.get(alias)
            if not isinstance(record, Mapping):
                continue
            prefix = f"checkpoint/{alias}"
            update_count = _as_scalar(record.get("update_count"))
            policy_version = _as_scalar(record.get("policy_version"))
            metric_value = _as_scalar(record.get("metric_value"))
            if update_count is not None:
                self._log_scalar(f"{prefix}/update_count", update_count, step)
            if policy_version is not None:
                self._log_scalar(f"{prefix}/policy_version", policy_version, step)
            if metric_value is not None:
                self._log_scalar(f"{prefix}/metric_value", metric_value, step)
            self.log_text(f"{prefix}/record", dict(record), step=step)
        self._logged_checkpoint_tracker_payloads.add(dedupe_key)

    def log_periodic_dev_eval(self, summary_payload: Mapping[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        self.log_text("dev_eval/summary", summary_payload, step=step)
        for tag, value in _iter_numeric_mapping(summary_payload, prefix="dev_eval"):
            self._log_scalar(tag, value, step)

    def log_final_eval_summary(self, summary_payload: Mapping[str, Any], *, step: int = 0) -> None:
        if not self.enabled:
            return
        self.log_text("eval/final/summary", summary_payload, step=step)
        policy_ids = [str(policy_id) for policy_id in cast(Sequence[Any], summary_payload.get("policy_ids", []))]
        self._log_scalar("eval/final/policy_count", float(len(policy_ids)), step)
        self._log_scalar(
            "eval/final/matchup_count",
            float(len(cast(list[Any], summary_payload.get("matchups", [])))),
            step,
        )
        posterior = summary_payload.get("posterior_samples", {})
        if isinstance(posterior, Mapping):
            sample_count = _as_scalar(posterior.get("sample_count"))
            if sample_count is not None:
                self._log_scalar("eval/final/posterior_sample_count", sample_count, step)
        matrices = summary_payload.get("matrices", {})
        if isinstance(matrices, Mapping):
            for matrix_name, matrix_payload in matrices.items():
                if not isinstance(matrix_payload, Mapping):
                    continue
                self._log_matrix_scalars(
                    prefix=f"eval/final/{_tag_part(matrix_name)}",
                    matrix_payload=matrix_payload,
                    step=step,
                )
                if matrix_name in {"mean", "ci_half_width", "games", "truncations", "prob_gt_half"}:
                    self._log_matrix_figure(
                        tag=f"eval/final/{_tag_part(matrix_name)}_heatmap",
                        title=f"Final Eval {matrix_name}",
                        matrix_payload=matrix_payload,
                        step=step,
                    )

    def log_metagame_summary(self, summary_payload: Mapping[str, Any], *, metagame_dir: Path, step: int = 0) -> None:
        if not self.enabled:
            return
        self.log_text("eval/metagame/summary", summary_payload, step=step)
        policy_ids = [str(policy_id) for policy_id in cast(Sequence[Any], summary_payload.get("policy_ids", []))]
        self._log_scalar("eval/metagame/policy_count", float(len(policy_ids)), step)
        sample_count = _as_scalar(summary_payload.get("sample_count"))
        if sample_count is not None:
            self._log_scalar("eval/metagame/sample_count", sample_count, step)
        for metric_name, relative_path in (
            ("nash_mixture", Path("S0") / "nash" / "mixture_mean.csv"),
            ("alpharank_stationary", Path("S0") / "alpharank" / "stationary_mean.csv"),
        ):
            csv_path = Path(metagame_dir) / relative_path
            if not csv_path.is_file():
                continue
            rows = _policy_metric_rows(csv_path)
            for policy_id, value in rows:
                self._log_scalar(f"eval/metagame/{metric_name}/{_tag_part(policy_id)}", value, step)
            self._log_policy_bar_chart(
                tag=f"eval/metagame/{metric_name}_bar",
                title=metric_name.replace("_", " ").title(),
                rows=rows,
                step=step,
            )

    def log_paper_readiness(self, summary_payload: Mapping[str, Any], *, step: int = 0) -> None:
        if not self.enabled:
            return
        self.log_text("eval/readiness/summary", summary_payload, step=step)
        self._log_scalar("eval/readiness/passed", 1.0 if bool(summary_payload.get("passed", False)) else 0.0, step)
        self._log_scalar(
            "eval/readiness/alarms_count", float(len(cast(list[Any], summary_payload.get("alarms", [])))), step
        )
        checks = summary_payload.get("checks", {})
        if isinstance(checks, Mapping):
            for check_name, check_payload in checks.items():
                if isinstance(check_payload, Mapping):
                    for tag, value in _iter_numeric_mapping(
                        check_payload, prefix=f"eval/readiness/checks/{_tag_part(check_name)}"
                    ):
                        self._log_scalar(tag, value, step)

    def log_text(self, tag: str, payload: Any, *, step: int = 0) -> None:
        if not self.enabled or self._writer is None:
            return
        dedupe_key = f"{tag}@{step}"
        if dedupe_key in self._logged_text_tags:
            return
        self._writer.add_text(tag, _json_markdown(payload), global_step=step)
        self._logged_text_tags.add(dedupe_key)

    def _training_tag_for_key(self, key: str) -> str:
        if key.startswith("vtrace_"):
            return f"vtrace/{_tag_part(key.removeprefix('vtrace_'))}"
        if key.startswith("throughput_"):
            return f"throughput/{_tag_part(key.removeprefix('throughput_'))}"
        if key.startswith("queue_") or key.startswith("policy_version_lag_") or key.startswith("snapshot_"):
            return f"runtime/{_tag_part(key)}"
        if key.startswith("actor_env_steps_"):
            return f"runtime/{_tag_part(key)}"
        if key.startswith("collector_"):
            return f"runtime/{_tag_part(key)}"
        if key.startswith("pfsp_"):
            return f"league/{_tag_part(key)}"
        if key in {"loss", "policy_loss", "value_loss", "actor_loss", "entropy", "grad_norm", "kl_divergence"}:
            return f"train/{_tag_part(key)}"
        if key in {"approx_kl", "clip_fraction", "explained_variance", "policy_train_fraction", "ppo_epochs_completed"}:
            return f"train/{_tag_part(key)}"
        return f"metrics/{_tag_part(key)}"

    def _log_scalar(self, tag: str, value: float, step: int, *, wall_clock_seconds: float | None = None) -> None:
        if not self.enabled or self._writer is None or not math.isfinite(value):
            return
        walltime = (
            None
            if wall_clock_seconds is None
            else time.time() - max(0.0, float(wall_clock_seconds)) + float(wall_clock_seconds)
        )
        self._writer.add_scalar(tag, value, global_step=int(step), walltime=walltime)

    def _log_matrix_scalars(self, *, prefix: str, matrix_payload: Mapping[str, Any], step: int) -> None:
        policy_ids = matrix_payload.get("policy_ids")
        values = matrix_payload.get("values")
        if not isinstance(policy_ids, Sequence) or isinstance(policy_ids, (str, bytes)):
            return
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return
        for row_index, row in enumerate(values):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                continue
            if row_index >= len(policy_ids):
                break
            focal_policy = _tag_part(policy_ids[row_index])
            for column_index, cell in enumerate(row):
                if column_index >= len(policy_ids):
                    break
                scalar = _as_scalar(cell)
                if scalar is None:
                    continue
                opponent_policy = _tag_part(policy_ids[column_index])
                self._log_scalar(f"{prefix}/{focal_policy}__vs__{opponent_policy}", scalar, step)

    def _log_matrix_figure(self, *, tag: str, title: str, matrix_payload: Mapping[str, Any], step: int) -> None:
        if not self.enabled or self._writer is None:
            return
        policy_ids = matrix_payload.get("policy_ids")
        matrix = _matrix_array(matrix_payload.get("values"))
        if matrix is None or not isinstance(policy_ids, Sequence) or isinstance(policy_ids, (str, bytes)):
            return
        labels = [_tag_part(policy_id) for policy_id in policy_ids]
        figure, axis = plt.subplots(figsize=(max(4.0, 0.8 * len(labels)), max(3.5, 0.8 * len(labels))))
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticklabels(labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        self._writer.add_figure(tag, figure, global_step=step, close=True)

    def _log_policy_bar_chart(
        self,
        *,
        tag: str,
        title: str,
        rows: Sequence[tuple[str, float]],
        step: int,
    ) -> None:
        if not self.enabled or self._writer is None or not rows:
            return
        labels = [_tag_part(policy_id) for policy_id, _value in rows]
        values = [float(value) for _policy_id, value in rows]
        figure, axis = plt.subplots(figsize=(max(4.0, 0.9 * len(labels)), 3.5))
        axis.bar(range(len(labels)), values, color="#2D6CDF")
        axis.set_title(title)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom")
        figure.tight_layout()
        self._writer.add_figure(tag, figure, global_step=step, close=True)


def tensorboard_unavailable_reason() -> str | None:
    if _SummaryWriterClass is not None:
        return None
    if _SUMMARY_WRITER_IMPORT_ERROR is None:
        return "SummaryWriter is unavailable"
    return str(_SUMMARY_WRITER_IMPORT_ERROR)
