"""IMPALA learner helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.learners.vtrace import VTraceTargets, compute_vtrace_metrics
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.training_logger import TrainingLogger, TrainingMetrics


VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    flat_rhos = np.asarray(result.rhos, dtype=np.float64).reshape(-1)
    if flat_rhos.size == 0:
        raise ValueError("result.rhos must not be empty")

    metrics = {
        f"vtrace_rho_p{percentile}": float(np.percentile(flat_rhos, percentile))
        for percentile in VTRACE_RHO_PERCENTILES
    }
    metrics["vtrace_rho_clip_rate"] = float(np.mean(flat_rhos > rho_bar))
    metrics["vtrace_c_clip_rate"] = float(np.mean(flat_rhos > c_bar))
    return metrics


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


@dataclass(slots=True)
class ImpalaLearner:
    learning_rate: float = 2e-4
    checkpoint_dir: Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: Path | None = None
    logging_interval_updates: int = 100
    vtrace_rho_bar: float = 2.4
    vtrace_c_bar: float = 1.0
    pass_action_id: int | None = None

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)

    def update(self, batch: Any) -> dict[str, float]:
        self.update_count += 1
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size

        elapsed = time.time() - self.start_time
        throughput_samples_per_sec = self.total_samples_processed / max(elapsed, 1e-6)
        throughput_updates_per_sec = self.update_count / max(elapsed, 1e-6)

        if self.checkpoint_dir and self.update_count % self.checkpoint_interval_updates == 0:
            self.policy_version += 1
            self._save_checkpoint()

        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(
                batch,
                throughput_samples_per_sec=throughput_samples_per_sec,
                throughput_updates_per_sec=throughput_updates_per_sec,
            )
            self.last_log_time = time.time()
            self.last_log_update = self.update_count

        metrics = {
            "loss": 0.0,
            "throughput_samples_per_sec": throughput_samples_per_sec,
            "throughput_updates_per_sec": throughput_updates_per_sec,
        }
        vtrace_result = _batch_value(batch, "vtrace_result")
        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
            c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))
        return metrics

    def _batch_size(self, batch: Any) -> int:
        if isinstance(batch, dict):
            for key in ("rewards", "actions", "logits"):
                if key in batch:
                    return int(np.asarray(batch[key]).size)
        return 1

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.update_count}.pt"
        checkpoint_path.write_text(
            f"update_count: {self.update_count}\npolicy_version: {self.policy_version}\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def _log_metrics(
        self,
        batch: Any,
        *,
        throughput_samples_per_sec: float,
        throughput_updates_per_sec: float,
    ) -> None:
        if not self.logger:
            return

        vtrace_metrics = compute_vtrace_metrics(
            batch if isinstance(batch, dict) else {},
            rho_bar=self.vtrace_rho_bar,
            c_bar=self.vtrace_c_bar,
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = TrainingMetrics(
            update_count=self.update_count,
            wall_clock_seconds=elapsed,
            wall_clock_ms=int(elapsed * 1000),
            policy_version=self.policy_version,
            loss=0.0,
            throughput_samples_per_sec=throughput_samples_per_sec,
            throughput_updates_per_sec=throughput_updates_per_sec,
            vtrace_rho_mean=vtrace_metrics.rho_mean,
            vtrace_rho_p50=vtrace_metrics.rho_p50,
            vtrace_rho_p90=vtrace_metrics.rho_p90,
            vtrace_rho_p99=vtrace_metrics.rho_p99,
            vtrace_clip_rate=vtrace_metrics.clip_rate,
            vtrace_c_clipped_rate=vtrace_metrics.c_clipped_rate,
            kl_divergence=vtrace_metrics.kl_divergence,
            entropy=vtrace_metrics.entropy,
        )
        self.logger.log(metrics)

    def get_policy_version(self) -> int:
        return self.policy_version
