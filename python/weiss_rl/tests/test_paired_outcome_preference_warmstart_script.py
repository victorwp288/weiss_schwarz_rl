from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scripts.paired_outcome_preference_warmstart import (
    _parse_pair_role_selectors,
    _parse_pair_weights,
    _preference_pair_role_mask,
    _preference_pair_weight_matrix,
    _scale_optimizer_learning_rates,
    _serialize_pair_role_selectors,
)


def test_preference_warmstart_can_scale_optimizer_learning_rates() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 2e-4}, {"lr": 1e-4, "name": "head"}])

    summary = _scale_optimizer_learning_rates(optimizer, scale=0.25)

    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[1]["lr"] == 2.5e-5
    assert summary == {
        "scale": 0.25,
        "groups": [
            {"index": 0, "original_lr": 2e-4, "scaled_lr": 5e-5},
            {"index": 1, "original_lr": 1e-4, "scaled_lr": 2.5e-5},
        ],
    }


def test_preference_warmstart_parses_pair_weights() -> None:
    assert _parse_pair_weights(["9=8", "14=2.5"]) == {9: 8.0, 14: 2.5}

    with pytest.raises(ValueError, match="PAIR_ID=WEIGHT"):
        _parse_pair_weights(["9"])
    with pytest.raises(ValueError, match="finite and positive"):
        _parse_pair_weights(["9=0"])


def test_preference_warmstart_builds_pair_weight_matrix() -> None:
    pair_ids = np.asarray([[9, 9, 14], [-1, 3, 3]], dtype=np.int64)

    weights = _preference_pair_weight_matrix(pair_ids, {9: 8.0, 14: 2.0})

    np.testing.assert_allclose(weights, np.asarray([[8.0, 8.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32))


def test_preference_warmstart_parses_and_builds_pair_role_masks() -> None:
    selectors = _parse_pair_role_selectors(["9:preferred", "14:rejected", "15:all", "9:preferred"])
    assert selectors == ((9, 1), (14, 0), (15, None))
    assert _serialize_pair_role_selectors(selectors) == ["9:preferred", "14:rejected", "15:all"]

    pair_ids = np.asarray([[9, 9, 14], [14, 15, 3]], dtype=np.int64)
    roles = np.asarray([[1, 0, 0], [1, 0, 1]], dtype=np.int64)

    mask = _preference_pair_role_mask(pair_ids, roles, selectors)

    np.testing.assert_allclose(mask, np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32))


def test_preference_warmstart_rejects_bad_pair_role_selectors() -> None:
    with pytest.raises(ValueError, match="PAIR_ID:ROLE"):
        _parse_pair_role_selectors(["9=preferred"])
    with pytest.raises(ValueError, match="preferred, rejected, or all"):
        _parse_pair_role_selectors(["9:maybe"])
