from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from weiss_rl.learners.logging import (
    build_training_metrics,
    checkpoint_metadata_payload,
    custom_log_metrics,
    write_checkpoint_metadata,
)
from weiss_rl.learners.vtrace import VtraceMetrics


def test_checkpoint_metadata_payload_and_writer_preserve_contract(tmp_path: Path) -> None:
    assert checkpoint_metadata_payload(update_count=3, policy_version=2) == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "update_count": 3,
        "policy_version": 2,
    }
    assert write_checkpoint_metadata(checkpoint_dir=None, update_count=3, policy_version=2) is None

    path = write_checkpoint_metadata(checkpoint_dir=tmp_path / "checkpoints", update_count=3, policy_version=2)

    assert path == tmp_path / "checkpoints" / "checkpoint_metadata_3.json"
    assert path is not None
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "policy_version": 2,
        "update_count": 3,
    }


def test_custom_log_metrics_reports_vtrace_availability_p95_and_entropy() -> None:
    vtrace = VtraceMetrics(rho_mean=1.0, entropy=0.25)

    metrics = custom_log_metrics(
        {
            "vtrace_rho_mean": 1.25,
            "vtrace_rho_p95": 3.5,
            "policy_train_fraction": 0.75,
            "reward_abs_mean": 0.125,
            "target_abs_mean": 0.875,
            "chosen_pass_train_fraction": 0.625,
            "chosen_pass_train_advantage_mean": -0.25,
            "teacher_hand_accuracy": 0.5,
            "teacher_main_play_character_slot_accuracy": 0.875,
            "teacher_hand_loss": 0.75,
            "teacher_hand_supported_fraction": 1.0,
            "policy_anchor_coef_active": 0.08,
            "policy_anchor_kl_mean": 0.12,
            "policy_anchor_top_action_coef_active": 0.04,
            "policy_anchor_top_action_agreement": 0.9,
            "trajectory_retention_coef_active": 0.06,
            "trajectory_retention_valid_fraction": 0.125,
            "trajectory_retention_supported_fraction": 1.0,
            "trajectory_retention_rows": 16.0,
            "trajectory_retention_loss": 0.75,
            "trajectory_retention_weighted_loss": 0.045,
            "trajectory_retention_logp_mean": -0.75,
            "trajectory_bc_replay_aux_updates": 1.0,
            "trajectory_bc_replay_focus_group_hard_negative_lossstate_repair_batch_episodes": 1.0,
            "trajectory_bc_replay_loss": 0.375,
            "paired_swing_replay_aux_updates": 1.0,
            "paired_swing_replay_paired_swing_loss": 0.125,
        },
        vtrace,
    )

    assert metrics == {
        "chosen_pass_train_advantage_mean": -0.25,
        "chosen_pass_train_fraction": 0.625,
        "policy_train_fraction": 0.75,
        "reward_abs_mean": 0.125,
        "target_abs_mean": 0.875,
        "teacher_hand_accuracy": 0.5,
        "teacher_main_play_character_slot_accuracy": 0.875,
        "teacher_hand_loss": 0.75,
        "teacher_hand_supported_fraction": 1.0,
        "policy_anchor_coef_active": 0.08,
        "policy_anchor_kl_mean": 0.12,
        "policy_anchor_top_action_coef_active": 0.04,
        "policy_anchor_top_action_agreement": 0.9,
        "trajectory_retention_coef_active": 0.06,
        "trajectory_retention_valid_fraction": 0.125,
        "trajectory_retention_supported_fraction": 1.0,
        "trajectory_retention_rows": 16.0,
        "trajectory_retention_loss": 0.75,
        "trajectory_retention_weighted_loss": 0.045,
        "trajectory_retention_logp_mean": -0.75,
        "trajectory_bc_replay_aux_updates": 1.0,
        "trajectory_bc_replay_focus_group_hard_negative_lossstate_repair_batch_episodes": 1.0,
        "trajectory_bc_replay_loss": 0.375,
        "paired_swing_replay_aux_updates": 1.0,
        "paired_swing_replay_paired_swing_loss": 0.125,
        "vtrace_batch_metrics_available": 1.0,
        "vtrace_entropy": 0.25,
        "vtrace_rho_mean": 1.25,
        "vtrace_rho_p95": 3.5,
    }

    unavailable = custom_log_metrics({}, VtraceMetrics(rho_mean=math.nan, entropy=math.nan))
    assert unavailable == {"vtrace_batch_metrics_available": 0.0}


def test_build_training_metrics_uses_update_overrides_and_vtrace_fallbacks() -> None:
    vtrace = VtraceMetrics(
        rho_mean=1.0,
        rho_p50=2.0,
        rho_p90=3.0,
        rho_p99=4.0,
        clip_rate=0.5,
        c_clipped_rate=0.25,
        kl_divergence=0.125,
        entropy=0.75,
    )

    metrics = build_training_metrics(
        update_metrics={
            "loss": 10.0,
            "throughput_samples_per_sec": 20.0,
            "throughput_updates_per_sec": 30.0,
            "vtrace_rho_mean": 35.0,
            "vtrace_rho_p50": 40.0,
            "vtrace_rho_p95": 50.0,
            "vtrace_rho_clip_rate": 0.75,
            "value_loss": 60.0,
            "policy_loss": 70.0,
            "entropy": 80.0,
        },
        vtrace_metrics=vtrace,
        update_count=5,
        policy_version=6,
        elapsed_seconds=1.234,
    )

    record = asdict(metrics)
    assert record["update_count"] == 5
    assert record["policy_version"] == 6
    assert record["wall_clock_ms"] == 1234
    assert record["loss"] == 10.0
    assert record["throughput_samples_per_sec"] == 20.0
    assert record["throughput_updates_per_sec"] == 30.0
    assert record["vtrace_rho_mean"] == 35.0
    assert record["vtrace_rho_p50"] == 40.0
    assert record["vtrace_rho_p90"] == 3.0
    assert record["vtrace_rho_p99"] == 4.0
    assert record["vtrace_clip_rate"] == 0.75
    assert record["vtrace_c_clipped_rate"] == 0.25
    assert record["kl_divergence"] == 0.125
    assert record["value_loss"] == 60.0
    assert record["actor_loss"] == 70.0
    assert record["entropy"] == 80.0
    assert record["custom_metrics"] == {
        "vtrace_batch_metrics_available": 1.0,
        "vtrace_entropy": 0.75,
        "vtrace_rho_mean": 35.0,
        "vtrace_rho_p95": 50.0,
    }
