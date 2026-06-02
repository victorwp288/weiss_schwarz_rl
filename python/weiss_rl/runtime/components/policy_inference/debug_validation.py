"""Debug validators for packed legal-action runtime paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.runtime.components.legal_batching import slice_packed_rows


def validate_sampled_packed_actions(
    *,
    source_label: str,
    row_indices: np.ndarray,
    action_subset: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    pass_action_id: int,
) -> None:
    subset_ids, subset_offsets = slice_packed_rows(legal_ids, legal_offsets, row_indices)
    for local_index, env_row in enumerate(row_indices.tolist()):
        start = int(subset_offsets[local_index])
        stop = int(subset_offsets[local_index + 1])
        row_ids = np.asarray(subset_ids[start:stop], dtype=np.uint32)
        action = int(np.asarray(action_subset, dtype=np.int64)[local_index])
        if row_ids.size == 0:
            if action != int(pass_action_id):
                raise ValueError(
                    f"debug invalid sampled packed action source={source_label} env_row={env_row} "
                    f"action={action} expected_pass={int(pass_action_id)}"
                )
            continue
        position = int(np.searchsorted(row_ids, action))
        is_legal = position < int(row_ids.size) and int(row_ids[position]) == action
        if not is_legal:
            preview = row_ids[: min(32, int(row_ids.size))].tolist()
            raise ValueError(
                f"debug invalid sampled packed action source={source_label} env_row={env_row} "
                f"local_row={local_index} action={action} legal_count={int(row_ids.size)} "
                f"legal_preview={preview}"
            )


def validate_env_step_packed_actions(
    *,
    source_label: str,
    actions: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    env_batch: Any,
) -> None:
    if env_batch is None or env_batch.ids_offsets is None:
        raise RuntimeError(f"debug env-step validation requires ids_offsets batch for {source_label}")
    env_legal_ids, env_legal_offsets = env_batch.ids_offsets
    action_array = np.asarray(actions, dtype=np.int64)
    local_ids_array = np.asarray(legal_ids, dtype=np.uint32)
    local_offsets_array = np.asarray(legal_offsets, dtype=np.uint32)
    env_ids_array = np.asarray(env_legal_ids, dtype=np.uint32)
    env_offsets_array = np.asarray(env_legal_offsets, dtype=np.uint32)
    if local_offsets_array.shape != env_offsets_array.shape:
        raise ValueError(
            f"debug env-step legality offsets shape mismatch source={source_label} "
            f"local_shape={tuple(local_offsets_array.shape)} env_shape={tuple(env_offsets_array.shape)}"
        )
    for env_index, action in enumerate(action_array.tolist()):
        local_start = int(local_offsets_array[env_index])
        local_stop = int(local_offsets_array[env_index + 1])
        env_start = int(env_offsets_array[env_index])
        env_stop = int(env_offsets_array[env_index + 1])
        local_row_ids = local_ids_array[local_start:local_stop]
        env_row_ids = env_ids_array[env_start:env_stop]
        local_position = int(np.searchsorted(local_row_ids, action))
        env_position = int(np.searchsorted(env_row_ids, action))
        local_legal = local_position < int(local_row_ids.size) and int(local_row_ids[local_position]) == action
        env_legal = env_position < int(env_row_ids.size) and int(env_row_ids[env_position]) == action
        if local_legal and env_legal and np.array_equal(local_row_ids, env_row_ids):
            continue
        local_preview = local_row_ids[: min(32, int(local_row_ids.size))].tolist()
        env_preview = env_row_ids[: min(32, int(env_row_ids.size))].tolist()
        raise ValueError(
            f"debug env-step legality mismatch source={source_label} env_row={env_index} action={action} "
            f"local_legal={bool(local_legal)} env_legal={bool(env_legal)} "
            f"local_count={int(local_row_ids.size)} env_count={int(env_row_ids.size)} "
            f"local_preview={local_preview} env_preview={env_preview}"
        )
