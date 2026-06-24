from __future__ import annotations

import numpy as np
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.runtime.components.actions.legal_meta import (
    action_catalog_indices,
    ensure_legal_action_meta,
    legal_action_meta_from_ids,
)


def _catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 41,
                "pass_action_id": 40,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 9},
                    {"name": "main_move", "base": 19, "count": 20},
                    {"name": "climax_play", "base": 39, "count": 1},
                    {"name": "pass", "base": 40, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_legal_action_meta_from_ids_encodes_family_and_action_slots() -> None:
    catalog = _catalog()
    family_index, attack_type_index = action_catalog_indices(catalog)

    meta = legal_action_meta_from_ids(
        np.array([7, 14, 25, 39, 40], dtype=np.uint32),
        action_catalog=catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=2,
    )

    unused = np.iinfo(np.uint16).max
    assert meta is not None
    assert meta.dtype == np.uint16
    assert meta.shape == (5, 4)
    np.testing.assert_array_equal(
        meta,
        np.array(
            [
                [0, 1, 2, unused],
                [1, 1, 1, unused],
                [2, 1, 3, unused],
                [3, 0, unused, unused],
                [4, unused, unused, unused],
            ],
            dtype=np.uint16,
        ),
    )


def test_legal_action_meta_from_ids_preserves_wider_action_meta_width() -> None:
    catalog = _catalog()
    family_index, attack_type_index = action_catalog_indices(catalog)

    meta = legal_action_meta_from_ids(
        np.array([40], dtype=np.uint32),
        action_catalog=catalog,
        family_index=family_index,
        attack_type_index=attack_type_index,
        action_meta_width=6,
    )

    assert meta is not None
    assert meta.shape == (1, 6)
    assert meta[0, 0] == 4


def test_ensure_legal_action_meta_casts_existing_meta_without_rebuilding() -> None:
    existing = np.array([[1, 2, 3, 4]], dtype=np.int64)

    meta = ensure_legal_action_meta(
        np.array([7], dtype=np.uint32),
        existing,
        build_meta=lambda _ids: (_ for _ in ()).throw(AssertionError("should not rebuild")),
    )

    assert meta is not None
    assert meta.dtype == np.uint16
    np.testing.assert_array_equal(meta, np.array([[1, 2, 3, 4]], dtype=np.uint16))


def test_legal_action_meta_from_ids_returns_none_without_catalog() -> None:
    assert (
        legal_action_meta_from_ids(
            np.array([7], dtype=np.uint32),
            action_catalog=None,
            family_index={},
            attack_type_index={},
            action_meta_width=4,
        )
        is None
    )
