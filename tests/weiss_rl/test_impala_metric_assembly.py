from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.loss_metrics import build_impala_loss_metrics
from weiss_rl.learners.impala.metrics_assembly import ImpalaMetricAssemblyRequest, assemble_impala_loss_metrics
from weiss_rl.learners.structured_policy_metrics import summarize_structured_policy_metrics

from .impala_test_support import _packed_ids_from_mask, _structured_metric_catalog


def test_assemble_impala_loss_metrics_preserves_base_metric_inputs_and_backfill_fields() -> None:
    batch = {
        "terminal_outcome_backfill_count": 3,
        "terminal_outcome_backfill_total_micros": 12.5,
        "terminal_outcome_trace_backfill_count": 2,
        "terminal_outcome_trace_backfill_total_micros": 7.25,
    }

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(1.5),
            policy_loss=torch.tensor(0.25),
            value_loss=torch.tensor(2.0),
            entropy_mean=torch.tensor(0.125),
            entropy_scope="candidate",
            loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
            value_loss_mask=torch.as_tensor([[1.0], [1.0]], dtype=torch.float32),
            actions=torch.as_tensor([[0], [1]], dtype=torch.long),
            action_logp=torch.as_tensor([[-0.2], [-0.3]], dtype=torch.float32),
            behavior_logp_for_mask=torch.as_tensor([[-0.1], [-0.5]], dtype=torch.float32),
            rewards_for_metrics=torch.as_tensor([[1.0], [-1.0]], dtype=torch.float32),
            advantages=torch.as_tensor([[0.5], [-0.25]], dtype=torch.float32),
            targets=torch.as_tensor([[1.25], [-0.75]], dtype=torch.float32),
            rhos_for_metrics=torch.as_tensor([[1.0], [2.0]], dtype=torch.float32),
            rho_bar=1.5,
            c_bar=1.25,
            action_catalog=object(),
            pass_action_id=1,
            trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
            policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
            teacher_metrics={"teacher_aux_loss": 0.5},
            emit_structured_metrics=True,
            batch=batch,
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda _name, _duration: pytest.fail("non-structured catalog must not summarize"),
    )

    assert metrics["loss"] == pytest.approx(1.5)
    assert metrics["policy_loss"] == pytest.approx(0.25)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert metrics["entropy"] == pytest.approx(0.125)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(3.0)
    assert metrics["terminal_outcome_backfill_total_micros"] == pytest.approx(12.5)
    assert metrics["terminal_outcome_trace_backfill_count"] == pytest.approx(2.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(7.25)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_aux_loss"] == pytest.approx(0.5)
    assert "structured_exact_action_concentration" not in metrics


def test_assemble_impala_loss_metrics_merges_structured_summary_with_dense_fallback() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 7, action_catalog.pass_action_id]] = True
    legal_mask[1, 0, [4, 7, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, action_catalog.pass_action_id] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, action_catalog.pass_action_id] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = {"marker": True}

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(0.0),
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy_mean=torch.tensor(0.0),
            entropy_scope="family",
            loss_mask=torch.ones((2, 1), dtype=torch.float32),
            value_loss_mask=torch.ones((2, 1), dtype=torch.float32),
            actions=torch.as_tensor([[0], [4]], dtype=torch.long),
            action_logp=torch.zeros((2, 1), dtype=torch.float32),
            behavior_logp_for_mask=None,
            rewards_for_metrics=torch.zeros((2, 1), dtype=torch.float32),
            advantages=torch.zeros((2, 1), dtype=torch.float32),
            targets=torch.zeros((2, 1), dtype=torch.float32),
            rhos_for_metrics=torch.ones((2, 1), dtype=torch.float32),
            rho_bar=1.0,
            c_bar=1.0,
            action_catalog=action_catalog,
            pass_action_id=action_catalog.pass_action_id,
            emit_structured_metrics=True,
            logits=logits,
            legal_mask=None,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=action_catalog.action_space_size,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    expected_structured = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    assert resolver_calls == [(batch, torch.Size((2, 1)), action_catalog.action_space_size)]
    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["structured_exact_action_concentration"] == pytest.approx(
        expected_structured["structured_exact_action_concentration"]
    )
    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(
        expected_structured["structured_main_move_0_2_top1_rate"]
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]


def test_impala_loss_metrics_builder_preserves_training_diagnostics_contract() -> None:
    metrics = build_impala_loss_metrics(
        total_loss=torch.tensor(2.0),
        policy_loss=torch.tensor(0.5),
        value_loss=torch.tensor(1.25),
        entropy_mean=torch.tensor(0.125),
        entropy_scope="family",
        loss_mask=torch.tensor([[1.0], [0.0]]),
        value_loss_mask=torch.tensor([[1.0], [1.0]]),
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        action_logp=torch.tensor([[-0.2], [-0.3]]),
        behavior_logp_for_mask=torch.tensor([[-0.5], [-0.3]]),
        rewards_for_metrics=torch.tensor([[0.0], [1.0]]),
        advantages=torch.tensor([[1.5], [-0.25]]),
        targets=torch.tensor([[0.25], [-0.5]]),
        rhos_for_metrics=torch.tensor([[2.0], [4.0]]),
        rho_bar=3.0,
        c_bar=1.5,
        action_catalog=None,
        pass_action_id=1,
        terminal_outcome_backfill_count=7,
        terminal_outcome_backfill_total_micros=11,
        terminal_outcome_trace_backfill_count=13,
        terminal_outcome_trace_backfill_total_micros=17,
        trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
        policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
        teacher_metrics={"teacher_valid_fraction": 0.5},
    )

    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["vtrace_rho_mean"] == pytest.approx(3.0)
    assert metrics["vtrace_train_rho_mean"] == pytest.approx(2.0)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)
    assert metrics["target_behavior_logp_delta_abs_mean"] == pytest.approx(0.15)
    assert metrics["target_behavior_train_logp_delta_abs_mean"] == pytest.approx(0.3)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(7.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(17.0)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.5)
