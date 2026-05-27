from __future__ import annotations

import pytest
import torch

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.impala_learner import (
    summarize_structured_policy_metrics as impala_summarize_structured_policy_metrics,
)
from weiss_rl.learners.structured_policy_metrics import summarize_structured_policy_metrics


def _structured_metric_catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 26,
                "pass_action_id": 25,
                "constants": [["MAX_HAND", 1], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 5},
                    {"name": "main_move", "base": 5, "count": 20},
                    {"name": "pass", "base": 25, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0]],
            }
        }
    )


def test_structured_policy_metrics_direct_helper_matches_impala_wrapper() -> None:
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
    legal_mask[1, 0, [4, main_move_02_action, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, main_move_02_action] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, main_move_02_action] = 0.0
    logits[1, 0, 25] = 0.5

    direct_metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    wrapper_metrics = impala_summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)

    assert impala_summarize_structured_policy_metrics is not summarize_structured_policy_metrics
    assert wrapper_metrics == pytest.approx(direct_metrics)
    assert direct_metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(0.5)
    assert 0.0 < direct_metrics["structured_exact_action_concentration"] <= 1.0


def test_structured_policy_metrics_factorized_family_path() -> None:
    action_catalog = _structured_metric_catalog()
    family_log_probs = torch.log_softmax(torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]]]), dim=-1)

    metrics = summarize_structured_policy_metrics(
        None,
        None,
        action_catalog=action_catalog,
        factorized_family_log_probs=family_log_probs,
    )

    assert metrics["structured_exact_action_concentration"] > 0.0
    assert metrics["structured_main_play_character_mass"] > 0.0
    assert metrics["structured_main_move_mass"] > 0.0
    assert metrics["structured_pass_mass"] > 0.0
