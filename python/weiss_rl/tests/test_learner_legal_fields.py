from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.learners.legal_fields import (
    has_legal_actions,
    require_actions,
    require_legal_mask,
    require_obs,
    resolve_legal_mask,
    resolve_packed_legal_actions_with_meta,
)


def _reference() -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32)


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def test_require_obs_actions_and_legal_mask_validate_shapes_and_dtypes() -> None:
    obs = require_obs(np.zeros((2, 3, 4), dtype=np.float64), reference=_reference())
    assert obs.dtype == torch.float32
    assert obs.shape == (2, 3, 4)
    with pytest.raises(ValueError, match="obs must be 3D"):
        require_obs(np.zeros((2, 3), dtype=np.float32), reference=_reference())

    actions = require_actions([[1, 2], [3, 4]], expected_shape=torch.Size((2, 2)), reference=_reference())
    assert actions.dtype == torch.long
    with pytest.raises(ValueError, match=r"actions must have shape \(2, 2\), got \(4,\)"):
        require_actions([1, 2, 3, 4], expected_shape=torch.Size((2, 2)), reference=_reference())

    legal_mask = require_legal_mask(
        np.ones((2, 2, 5), dtype=np.uint8),
        expected_shape=torch.Size((2, 2)),
        reference=_reference(),
    )
    assert legal_mask.dtype == torch.bool
    with pytest.raises(ValueError, match=r"legal_mask must have shape \(2, 2, 'action'\), got \(2, 5\)"):
        require_legal_mask(np.ones((2, 5), dtype=np.uint8), expected_shape=torch.Size((2, 2)), reference=_reference())


def test_has_legal_actions_accepts_all_supported_representations() -> None:
    assert has_legal_actions({"legal_actions": object()}, batch_value=_batch_value)
    assert has_legal_actions({"legal_mask": object()}, batch_value=_batch_value)
    assert has_legal_actions({"legal_ids": [1], "legal_offsets": [0, 1]}, batch_value=_batch_value)
    assert not has_legal_actions({"legal_ids": [1]}, batch_value=_batch_value)
    assert has_legal_actions(SimpleNamespace(legal_mask=np.ones((1, 1, 2))), batch_value=_batch_value)


def test_resolve_legal_mask_from_dense_legal_action_batch_and_packed_ids() -> None:
    expected_shape = torch.Size((2, 1))
    dense = resolve_legal_mask(
        {"legal_mask": np.asarray([[[1, 0, 1]], [[0, 1, 0]]], dtype=np.uint8)},
        expected_shape=expected_shape,
        action_dim=3,
        reference=_reference(),
        batch_value=_batch_value,
    )
    assert dense.tolist() == [[[True, False, True]], [[False, True, False]]]

    legal_actions = LegalActionBatch.from_packed(
        ids=np.asarray([0, 2, 1], dtype=np.uint32),
        offsets=np.asarray([0, 2, 3], dtype=np.int64),
    )
    from_legal_actions = resolve_legal_mask(
        {"legal_actions": legal_actions},
        expected_shape=expected_shape,
        action_dim=3,
        reference=_reference(),
        batch_value=_batch_value,
    )
    from_ids = resolve_legal_mask(
        {"legal_ids": np.asarray([0, 2, 1], dtype=np.uint32), "legal_offsets": np.asarray([0, 2, 3])},
        expected_shape=expected_shape,
        action_dim=3,
        reference=_reference(),
        batch_value=_batch_value,
    )

    assert from_legal_actions.tolist() == dense.tolist()
    assert from_ids.tolist() == dense.tolist()
    with pytest.raises(ValueError, match="batch must include either legal_actions"):
        resolve_legal_mask(
            {}, expected_shape=expected_shape, action_dim=3, reference=_reference(), batch_value=_batch_value
        )


def test_resolve_packed_legal_actions_with_meta_validates_offsets_and_structured_metadata() -> None:
    expected_shape = torch.Size((2, 1))
    meta = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.uint16)
    packed = resolve_packed_legal_actions_with_meta(
        {
            "legal_ids": np.asarray([0, 2, 1], dtype=np.uint32),
            "legal_offsets": np.asarray([0, 2, 3], dtype=np.int64),
            "legal_action_meta": meta,
        },
        expected_shape=expected_shape,
        reference=_reference(),
        batch_value=_batch_value,
        supports_legal_candidate_scoring=True,
    )

    assert packed is not None
    ids, offsets, resolved_meta = packed
    assert ids.tolist() == [0, 2, 1]
    assert offsets.tolist() == [0, 2, 3]
    assert resolved_meta is not None
    assert resolved_meta.tolist() == meta.tolist()

    legal_actions = LegalActionBatch.from_packed(
        ids=np.asarray([0, 2, 1], dtype=np.uint32),
        offsets=np.asarray([0, 2, 3], dtype=np.int64),
        meta=meta,
    )
    from_legal_actions = resolve_packed_legal_actions_with_meta(
        {"legal_actions": legal_actions},
        expected_shape=expected_shape,
        reference=_reference(),
        batch_value=_batch_value,
        supports_legal_candidate_scoring=True,
    )
    assert from_legal_actions is not None
    assert from_legal_actions[2] is not None
    assert from_legal_actions[2].tolist() == meta.tolist()

    assert (
        resolve_packed_legal_actions_with_meta(
            {},
            expected_shape=expected_shape,
            reference=_reference(),
            batch_value=_batch_value,
            supports_legal_candidate_scoring=False,
        )
        is None
    )
    with pytest.raises(ValueError, match=r"packed legal offsets must have shape \(3,\)"):
        resolve_packed_legal_actions_with_meta(
            {"legal_ids": [0], "legal_offsets": [0, 1]},
            expected_shape=expected_shape,
            reference=_reference(),
            batch_value=_batch_value,
            supports_legal_candidate_scoring=False,
        )
    with pytest.raises(ValueError, match="structured learner updates require packed legal action metadata"):
        resolve_packed_legal_actions_with_meta(
            {"legal_ids": [0], "legal_offsets": [0, 1, 1]},
            expected_shape=expected_shape,
            reference=_reference(),
            batch_value=_batch_value,
            supports_legal_candidate_scoring=True,
        )
