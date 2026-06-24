"""Helpers for updating training run report payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingRunEvidenceArtifact:
    path: str
    purpose: str
    required_for: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "purpose": self.purpose,
            "required_for": list(self.required_for),
        }


TRAINING_RUN_EVIDENCE_ARTIFACTS: tuple[TrainingRunEvidenceArtifact, ...] = (
    TrainingRunEvidenceArtifact(
        path="manifest.json",
        purpose="run identity, config hash, spec hash, and simulator provenance",
        required_for=("all retained runs", "paper readiness"),
    ),
    TrainingRunEvidenceArtifact(
        path="run_summary.json",
        purpose="reader-facing run status, policy selection, controls, and evidence pointers",
        required_for=("all retained runs",),
    ),
    TrainingRunEvidenceArtifact(
        path="determinism_report.json",
        purpose="seed, runtime, and reproducibility context for the run",
        required_for=("all retained runs", "paper readiness"),
    ),
    TrainingRunEvidenceArtifact(
        path="training/logs/training_metrics.jsonl",
        purpose="per-update learner, runtime, reward, checkpoint, and diagnostic metrics",
        required_for=("training evidence", "learning diagnostics"),
    ),
    TrainingRunEvidenceArtifact(
        path="training/checkpoints/checkpoint_tracker.json",
        purpose="latest, best, selected, and finalized checkpoint aliases",
        required_for=("checkpoint selection", "final evaluation"),
    ),
    TrainingRunEvidenceArtifact(
        path="training/snapshots/registry.json",
        purpose="policy snapshot registry used by promotion and final policy-set resolution",
        required_for=("policy selection", "final evaluation"),
    ),
    TrainingRunEvidenceArtifact(
        path="training/logs/periodic_dev_eval_summaries.json",
        purpose="training-time anchor scores used as checkpoint and selection evidence",
        required_for=("learning quality", "policy selection"),
    ),
    TrainingRunEvidenceArtifact(
        path="eval/final_eval/summary.json",
        purpose="retained fixed-panel evaluation summary and policy-selection metadata",
        required_for=("thesis evidence", "paper readiness"),
    ),
    TrainingRunEvidenceArtifact(
        path="paper_readiness_summary.json",
        purpose="run-tree contract result for paper-grade retained evidence",
        required_for=("paper readiness",),
    ),
)


def training_run_evidence_artifact_payload() -> list[dict[str, object]]:
    return [artifact.as_payload() for artifact in TRAINING_RUN_EVIDENCE_ARTIFACTS]


def policy_selection_report_payload(policy_set_selection_details: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact policy-selection block used by top-level run reports."""

    payload: dict[str, Any] = {
        "mode": policy_set_selection_details.get("mode", "unresolved"),
        "status": policy_set_selection_details.get("status", "unresolved"),
    }
    for key in ("version", "final_policy_set_size", "selected_policy_count", "missing_inputs", "source_paths"):
        if key in policy_set_selection_details:
            payload[key] = policy_set_selection_details[key]
    if "reason" in policy_set_selection_details:
        payload["reason"] = policy_set_selection_details["reason"]

    selection_trace = policy_set_selection_details.get("selection_trace")
    if isinstance(selection_trace, Sequence) and not isinstance(selection_trace, (str, bytes)):
        payload["selection_reasons"] = _selection_reason_rows(selection_trace)
    return payload


def _selection_reason_rows(selection_trace: Sequence[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_entry in selection_trace:
        if not isinstance(raw_entry, Mapping):
            continue
        policy_id = raw_entry.get("policy_id")
        reason = raw_entry.get("reason")
        if not isinstance(policy_id, str) or not isinstance(reason, str):
            continue
        rows.append({"policy_id": policy_id, "reason": reason})
    return rows


def training_controls_payload(training_config: Any | None) -> dict[str, str | bool | float] | None:
    """Return the run-report training controls payload for a configured training run."""

    if training_config is None:
        return None
    return {
        "profile_timers": bool(training_config.profile_timers),
        "torch_profiler": bool(training_config.torch_profiler),
        "structured_metrics_mode": str(training_config.structured_metrics_mode),
        "teacher_aux_mode": str(training_config.teacher_aux_mode),
        "fixed_opponent_backend": str(training_config.fixed_opponent_backend),
        "actor_policy_backend": str(training_config.actor_policy_backend),
        "actor_heuristic_fraction": float(training_config.actor_heuristic_fraction),
        "actor_heuristic_final_fraction": float(training_config.actor_heuristic_final_fraction),
        "actor_sampling_temperature": float(getattr(training_config, "actor_sampling_temperature", 1.0)),
        "train_on_heuristic_actor_rows": bool(training_config.train_on_heuristic_actor_rows),
    }


def profiling_enabled_message(training_config: Any) -> str | None:
    """Return the user-facing structured profiling message when profiling is active."""

    profile_timers = bool(training_config.profile_timers)
    torch_profiler = bool(training_config.torch_profiler)
    if not profile_timers and not torch_profiler:
        return None
    return (
        "Structured profiling enabled: "
        f"profile_timers={profile_timers} "
        f"torch_profiler={torch_profiler} "
        f"structured_metrics_mode={training_config.structured_metrics_mode} "
        f"teacher_aux_mode={training_config.teacher_aux_mode} "
        f"fixed_opponent_backend={training_config.fixed_opponent_backend}"
    )


def augment_run_summary_payload(
    payload: MutableMapping[str, Any],
    *,
    public_demo_enabled: bool,
    runtime_mode: str,
    policy_set_selection_details: Mapping[str, Any],
    training_config: Any | None,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    init_from_checkpoint_path: Path | None,
    resume_run_dir: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the run summary payload."""

    payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(runtime_mode)
    payload["run_evidence_artifacts"] = training_run_evidence_artifact_payload()
    payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    payload["policy_set_selection"] = policy_selection_report_payload(policy_set_selection_details)
    training_controls = training_controls_payload(training_config)
    if training_controls is not None:
        payload["training_controls"] = training_controls
    if b1_baseline_run_dir is not None:
        payload["b1_baseline_run_dir"] = b1_baseline_run_dir.resolve().as_posix()
    if seed_snapshot_run_dir is not None:
        payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.resolve().as_posix()
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.resolve().as_posix()
    if resume_checkpoint_path is not None:
        payload["resume"] = {
            "enabled": True,
            "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
            "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
        }
    return payload


def augment_determinism_payload(
    payload: MutableMapping[str, Any],
    *,
    public_demo_enabled: bool,
    runtime_mode: str,
    policy_set_selection_details: Mapping[str, Any],
    training_config: Any | None,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None,
    init_from_checkpoint_path: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the determinism report payload."""

    payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(runtime_mode)
    payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
    payload["policy_selection"] = policy_selection_report_payload(policy_set_selection_details)
    training_controls = training_controls_payload(training_config)
    if training_controls is not None:
        payload["training_controls"] = training_controls
    if b1_baseline_run_dir is not None:
        payload["b1_baseline_run_dir"] = b1_baseline_run_dir.resolve().as_posix()
    if seed_snapshot_run_dir is not None:
        payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.resolve().as_posix()
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.resolve().as_posix()
    if resume_checkpoint_path is not None:
        payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    return payload


def augment_environment_payload(
    payload: MutableMapping[str, Any],
    *,
    root: Path,
    argv: Sequence[str],
    hardware: Mapping[str, Any],
    init_from_checkpoint_path: Path | None,
    resume_checkpoint_path: Path | None,
) -> MutableMapping[str, Any]:
    """Apply train-entrypoint fields to the environment manifest payload."""

    payload["cwd"] = root.as_posix()
    payload["argv"] = list(argv)
    payload["hardware"] = hardware
    if init_from_checkpoint_path is not None:
        payload["init_from_checkpoint_path"] = init_from_checkpoint_path.as_posix()
    if resume_checkpoint_path is not None:
        payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
    return payload
