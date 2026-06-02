from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch

from weiss_rl.core.masking import select_argmax_from_legal_ids, select_argmax_from_mask
from weiss_rl.runtime.components.legal_batching import (
    optional_legal_action_meta,
    require_ids_offsets,
    require_mask,
    slice_packed_rows_with_meta,
)
from weiss_rl.runtime.components.opponents.central_heuristic_opponents import legal_action_ids_from_mask_rows
from weiss_rl.runtime.components.opponents.central_opponent_groups import CentralOpponentEntry
from weiss_rl.runtime.components.policy_inference.deterministic_logits import (
    write_deterministic_logits,
    write_deterministic_logits_from_packed,
)


class CentralSnapshotForwardBatch(NamedTuple):
    obs: np.ndarray
    actor: np.ndarray
    hidden: torch.Tensor


class CentralSnapshotModelOutputs(NamedTuple):
    logits: np.ndarray
    values: np.ndarray
    next_hidden: torch.Tensor


def build_central_snapshot_forward_batch(entries: list[CentralOpponentEntry]) -> CentralSnapshotForwardBatch:
    hidden = torch.cat([entry.actor.opponent_hidden[entry.row_indices] for entry in entries], dim=0)
    return CentralSnapshotForwardBatch(
        obs=np.concatenate([entry.obs_step[entry.row_indices] for entry in entries], axis=0),
        actor=np.concatenate([entry.actor_step[entry.row_indices] for entry in entries], axis=0),
        hidden=hidden,
    )


def run_central_snapshot_model(
    *,
    model: Any,
    entries: list[CentralOpponentEntry],
    lock: Any,
    device: torch.device,
    amp_enabled: bool,
) -> CentralSnapshotModelOutputs:
    forward_batch = build_central_snapshot_forward_batch(entries)
    with (
        lock,
        torch.inference_mode(),
        torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ),
    ):
        logits_tensor, value_tensor, next_hidden = model.forward_seat_aware(
            torch.as_tensor(forward_batch.obs, device=device),
            torch.as_tensor(forward_batch.actor, device=device, dtype=torch.long),
            forward_batch.hidden,
        )
    return CentralSnapshotModelOutputs(
        logits=logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
        values=value_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
        next_hidden=torch.as_tensor(next_hidden, device=device, dtype=forward_batch.hidden.dtype),
    )


def apply_central_snapshot_opponent_policy(
    *,
    policy_id: str,
    entries: list[CentralOpponentEntry],
    opponent_models: Mapping[str, Any],
    opponent_model_locks: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    action_selection: str,
    pass_action_id: int,
    action_dim: int,
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
) -> None:
    model = opponent_models.get(policy_id)
    if model is None:
        raise RuntimeError(f"missing opponent snapshot model for policy_id {policy_id!r}")
    snapshot_outputs = run_central_snapshot_model(
        model=model,
        entries=entries,
        lock=opponent_model_locks[policy_id],
        device=device,
        amp_enabled=amp_enabled,
    )
    apply_central_snapshot_outputs(
        entries=entries,
        outputs=snapshot_outputs,
        action_selection=action_selection,
        pass_action_id=pass_action_id,
        action_dim=action_dim,
        ensure_legal_action_meta=ensure_legal_action_meta,
    )


def apply_central_snapshot_outputs(
    *,
    entries: list[CentralOpponentEntry],
    outputs: CentralSnapshotModelOutputs,
    action_selection: str,
    pass_action_id: int,
    action_dim: int,
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
) -> None:
    offset = 0
    for entry in entries:
        count = int(entry.row_indices.shape[0])
        next_offset = offset + count
        entry.actor.opponent_hidden[entry.row_indices] = outputs.next_hidden[offset:next_offset]
        entry.values_out[entry.row_indices] = outputs.values[offset:next_offset]
        if entry.logits_out is not None:
            row_logits = outputs.logits[offset:next_offset]
            entry.logits_out[entry.row_indices] = row_logits
            if str(action_selection) == "argmax":
                _write_argmax_deterministic_logits(
                    entry=entry,
                    row_logits=row_logits,
                    pass_action_id=pass_action_id,
                    action_dim=action_dim,
                    ensure_legal_action_meta=ensure_legal_action_meta,
                )
        offset = next_offset


def _write_argmax_deterministic_logits(
    *,
    entry: CentralOpponentEntry,
    row_logits: np.ndarray,
    pass_action_id: int,
    action_dim: int,
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
) -> None:
    if entry.batch.ids_offsets is not None:
        legal_ids, legal_offsets = require_ids_offsets(entry.batch)
        subset_ids, subset_offsets, _subset_meta = slice_packed_rows_with_meta(
            legal_ids,
            legal_offsets,
            entry.row_indices,
            legal_action_meta=ensure_legal_action_meta(legal_ids, optional_legal_action_meta(entry.batch)),
        )
        chosen_actions = select_argmax_from_legal_ids(
            row_logits,
            subset_ids,
            subset_offsets,
            pass_action_id=pass_action_id,
        )
        write_deterministic_logits_from_packed(
            logits_out=entry.logits_out,
            row_indices=entry.row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
        return

    legal_mask = require_mask(entry.batch)
    chosen_actions = select_argmax_from_mask(
        row_logits,
        legal_mask[entry.row_indices],
        pass_action_id=pass_action_id,
    )
    write_deterministic_logits(
        logits_out=entry.logits_out,
        row_indices=entry.row_indices,
        chosen_actions=chosen_actions,
        legal_action_ids=legal_action_ids_from_mask_rows(legal_mask, entry.row_indices),
        action_dim=action_dim,
    )
