"""Apply heuristic fixed-opponent outputs for central collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.runtime.components.legal_batching import (
    optional_legal_action_meta,
    require_ids_offsets,
    require_mask,
)
from weiss_rl.runtime.components.opponents.central_heuristic_opponents import (
    build_central_packed_heuristic_batch,
    legal_action_ids_from_mask_rows,
    split_central_heuristic_entries,
)
from weiss_rl.runtime.components.opponents.central_opponent_groups import CentralOpponentEntry


def apply_central_heuristic_opponent_outputs(
    *,
    policy_id: str,
    entries: Sequence[CentralOpponentEntry],
    heuristic_policy: Any,
    fixed_opponent_backend: str,
    track_heuristic_hidden_state: bool,
    central_advance_actor_rows: Callable[..., None],
    heuristic_public_actions_from_ids: Callable[..., np.ndarray],
    heuristic_public_actions_from_mask: Callable[..., np.ndarray],
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
    maybe_debug_validate_sampled_packed_actions: Callable[..., None],
    write_deterministic_logits_from_packed: Callable[..., None],
    write_deterministic_logits: Callable[..., None],
) -> None:
    if track_heuristic_hidden_state:
        _advance_heuristic_hidden(
            entries=entries,
            central_advance_actor_rows=central_advance_actor_rows,
        )
    heuristic_entries = split_central_heuristic_entries(entries)
    if heuristic_entries.packed:
        if str(fixed_opponent_backend) == "simulator_native":
            _apply_simulator_native_packed_heuristic_entries(
                policy_id=policy_id,
                entries=heuristic_entries.packed,
                heuristic_policy=heuristic_policy,
                heuristic_public_actions_from_ids=heuristic_public_actions_from_ids,
                ensure_legal_action_meta=ensure_legal_action_meta,
                maybe_debug_validate_sampled_packed_actions=maybe_debug_validate_sampled_packed_actions,
                write_deterministic_logits_from_packed=write_deterministic_logits_from_packed,
            )
        else:
            _apply_batched_packed_heuristic_entries(
                entries=heuristic_entries.packed,
                heuristic_policy=heuristic_policy,
                ensure_legal_action_meta=ensure_legal_action_meta,
                write_deterministic_logits_from_packed=write_deterministic_logits_from_packed,
            )
    if heuristic_entries.mask:
        _apply_mask_heuristic_entries(
            policy_id=policy_id,
            entries=heuristic_entries.mask,
            heuristic_policy=heuristic_policy,
            heuristic_public_actions_from_mask=heuristic_public_actions_from_mask,
            write_deterministic_logits=write_deterministic_logits,
        )


def _advance_heuristic_hidden(
    *,
    entries: Sequence[CentralOpponentEntry],
    central_advance_actor_rows: Callable[..., None],
) -> None:
    if not entries:
        return
    central_advance_actor_rows(
        actors=[entry.actor for entry in entries],
        obs_steps=[entry.obs_step for entry in entries],
        actor_steps=[entry.actor_step for entry in entries],
        row_indices_by_actor=[entry.row_indices for entry in entries],
    )


def _apply_simulator_native_packed_heuristic_entries(
    *,
    policy_id: str,
    entries: Sequence[CentralOpponentEntry],
    heuristic_policy: Any,
    heuristic_public_actions_from_ids: Callable[..., np.ndarray],
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
    maybe_debug_validate_sampled_packed_actions: Callable[..., None],
    write_deterministic_logits_from_packed: Callable[..., None],
) -> None:
    profile_name = heuristic_public_profile_name_for_policy_id(policy_id)
    for entry in entries:
        legal_ids, legal_offsets = require_ids_offsets(entry.batch)
        chosen_actions = heuristic_public_actions_from_ids(
            actor=entry.actor,
            heuristic_policy=heuristic_policy,
            row_indices=entry.row_indices,
            obs_step=entry.obs_step,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
            legal_action_meta=ensure_legal_action_meta(legal_ids, optional_legal_action_meta(entry.batch)),
            profile_name=profile_name,
        )
        maybe_debug_validate_sampled_packed_actions(
            source_label=f"central:opponent:{policy_id}:heuristic",
            row_indices=entry.row_indices,
            action_subset=np.asarray(chosen_actions, dtype=np.int64),
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
        write_deterministic_logits_from_packed(
            logits_out=entry.logits_out,
            row_indices=entry.row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
        entry.values_out[entry.row_indices] = 0.0


def _apply_batched_packed_heuristic_entries(
    *,
    entries: Sequence[CentralOpponentEntry],
    heuristic_policy: Any,
    ensure_legal_action_meta: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None],
    write_deterministic_logits_from_packed: Callable[..., None],
) -> None:
    packed_batch = build_central_packed_heuristic_batch(
        entries,
        ensure_legal_action_meta=ensure_legal_action_meta,
    )
    packed_chosen_actions = heuristic_policy.choose_actions_from_meta_batch(
        packed_batch.obs_rows,
        packed_batch.legal_ids,
        packed_batch.legal_offsets,
        packed_batch.legal_action_meta,
    )
    offset = 0
    for entry, count in zip(entries, packed_batch.entry_counts, strict=True):
        legal_ids, legal_offsets = require_ids_offsets(entry.batch)
        chosen_actions = np.asarray(
            packed_chosen_actions[offset : offset + count],
            dtype=np.int64,
        )
        write_deterministic_logits_from_packed(
            logits_out=entry.logits_out,
            row_indices=entry.row_indices,
            chosen_actions=chosen_actions,
            legal_ids=legal_ids,
            legal_offsets=legal_offsets,
        )
        entry.values_out[entry.row_indices] = 0.0
        offset += count


def _apply_mask_heuristic_entries(
    *,
    policy_id: str,
    entries: Sequence[CentralOpponentEntry],
    heuristic_policy: Any,
    heuristic_public_actions_from_mask: Callable[..., np.ndarray],
    write_deterministic_logits: Callable[..., None],
) -> None:
    profile_name = heuristic_public_profile_name_for_policy_id(policy_id)
    for entry in entries:
        legal_mask = require_mask(entry.batch)
        chosen_actions = heuristic_public_actions_from_mask(
            actor=entry.actor,
            heuristic_policy=heuristic_policy,
            row_indices=entry.row_indices,
            obs_step=entry.obs_step,
            legal_mask=legal_mask,
            profile_name=profile_name,
        )
        write_deterministic_logits(
            logits_out=entry.logits_out,
            row_indices=entry.row_indices,
            chosen_actions=chosen_actions,
            legal_action_ids=legal_action_ids_from_mask_rows(legal_mask, entry.row_indices),
        )
        entry.values_out[entry.row_indices] = 0.0
