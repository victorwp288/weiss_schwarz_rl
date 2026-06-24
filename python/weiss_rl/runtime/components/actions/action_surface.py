"""Batch-facing action-surface guards for simulator decision quirks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from weiss_rl.runtime.components.actions.action_surface_mulligan_guard import (
    filter_mulligan_select_after_select_from_ids,
)
from weiss_rl.runtime.components.actions.action_surface_packed import (
    PackedActionSurfaceFilterResult,
    empty_filter_result,
)
from weiss_rl.runtime.components.actions.action_surface_pass_guards import (
    filter_main_move_only_rows_to_pass_from_ids,
    filter_pass_when_attack_available_from_ids,
)


def _batch_with_filtered_surface(
    batch: Any,
    result: PackedActionSurfaceFilterResult,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if result.filtered_actions <= 0:
        return batch, result
    return (
        replace(
            batch,
            ids_offsets=(result.legal_ids, result.legal_offsets),
            legal_action_meta=result.legal_action_meta,
        ),
        result,
    )


def filter_batch_mulligan_select_after_select(
    batch: Any,
    *,
    last_action_arg0_index: int,
    mulligan_select_family_id: int,
    mulligan_confirm_family_id: int,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        return batch, empty_filter_result()
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_mulligan_select_after_select_from_ids(
        obs=np.asarray(batch.obs),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        last_action_arg0_index=int(last_action_arg0_index),
        mulligan_select_family_id=int(mulligan_select_family_id),
        mulligan_confirm_family_id=int(mulligan_confirm_family_id),
    )
    return _batch_with_filtered_surface(batch, result)


def filter_batch_main_move_only_rows_to_pass(
    batch: Any,
    *,
    pass_action_id: int,
    main_move_family_id: int,
    allow_main_move_only_rows: np.ndarray | None = None,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        return batch, empty_filter_result()
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_main_move_only_rows_to_pass_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        pass_action_id=int(pass_action_id),
        main_move_family_id=int(main_move_family_id),
        allow_main_move_only_rows=allow_main_move_only_rows,
    )
    return _batch_with_filtered_surface(batch, result)


def filter_batch_pass_when_attack_available(
    batch: Any,
    *,
    pass_action_id: int,
    attack_family_id: int,
) -> tuple[Any, PackedActionSurfaceFilterResult]:
    if getattr(batch, "ids_offsets", None) is None:
        return batch, empty_filter_result()
    legal_ids, legal_offsets = batch.ids_offsets
    result = filter_pass_when_attack_available_from_ids(
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=getattr(batch, "legal_action_meta", None),
        pass_action_id=int(pass_action_id),
        attack_family_id=int(attack_family_id),
    )
    return _batch_with_filtered_surface(batch, result)


__all__ = [
    "PackedActionSurfaceFilterResult",
    "filter_batch_main_move_only_rows_to_pass",
    "filter_batch_mulligan_select_after_select",
    "filter_batch_pass_when_attack_available",
    "filter_main_move_only_rows_to_pass_from_ids",
    "filter_mulligan_select_after_select_from_ids",
    "filter_pass_when_attack_available_from_ids",
]
