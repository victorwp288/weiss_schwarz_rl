from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.support.structured_summary import (
    ImpalaStructuredSummaryRequest,
    compute_impala_structured_policy_summary,
)
from weiss_rl.learners.structured_policy_metrics import summarize_structured_policy_metrics

from .impala_test_support import _packed_ids_from_mask, _packed_meta_from_ids, _structured_metric_catalog


def test_summarize_structured_policy_metrics_reports_mainmove_pressure() -> None:
    action_catalog = _structured_metric_catalog()
    main_move_02_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (
            action_catalog.decode(action_id).family == "main_move"
            and action_catalog.decode(action_id).from_slot == 0
            and action_catalog.decode(action_id).to_slot == 2
        )
    )
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)

    legal_mask[0, 0, [0, main_move_02_action, 25]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, main_move_02_action] = 2.0
    logits[0, 0, 25] = 1.0

    legal_mask[1, 0, [0, main_move_02_action, 25]] = True
    logits[1, 0, 0] = 3.0
    logits[1, 0, main_move_02_action] = 0.0
    logits[1, 0, 25] = 1.0

    metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)

    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(0.5)
    assert 0.0 < metrics["structured_main_move_share_when_play_available"] < 1.0
    assert (
        metrics["structured_main_play_character_mass"]
        + metrics["structured_main_move_mass"]
        + metrics["structured_pass_mass"]
    ) == pytest.approx(1.0)
    assert 0.0 < metrics["structured_exact_action_concentration"] <= 1.0


def test_summarize_structured_policy_metrics_matches_packed_meta_path() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    packed_metrics = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_impala_structured_policy_summary_resolves_dense_mask_when_packed_meta_missing() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = object()

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=26,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    assert resolver_calls == [(batch, torch.Size((2, 1)), 26)]
    assert metrics == pytest.approx(
        summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]
    assert timings[0][1] >= 0.0


def test_compute_impala_structured_policy_summary_keeps_packed_meta_path_without_dense_mask() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                torch.as_tensor(packed_meta, dtype=torch.long),
            ),
            resolve_legal_mask=lambda _source_batch, _expected_shape, _action_dim: pytest.fail(
                "packed metadata path should not reconstruct a dense mask"
            ),
        ),
        record_timing_ms=lambda _name, _duration: None,
    )

    expected = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    assert metrics == pytest.approx(expected)
