"""Heuristic-public opponent action routing for :class:`weiss_rl.runtime.QueueRuntime`."""

from __future__ import annotations

from typing import Any

import numpy as np

from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID, heuristic_public_profile_name_for_policy_id
from weiss_rl.runtime.components.legal_batching import slice_packed_rows, slice_packed_rows_with_meta


class QueueRuntimeHeuristicPublicActionsMixin:
    def _heuristic_opponent_policy(self: Any, policy_id: str) -> HeuristicPublicPolicy | None:
        policy_key = str(policy_id)
        heuristic_policies = getattr(self, "_opponent_heuristic_policies", {})
        heuristic_policy = heuristic_policies.get(policy_key)
        if heuristic_policy is None and policy_key == HEURISTIC_PUBLIC_POLICY_ID:
            heuristic_policy = getattr(self, "_teacher_policy", None)
        spec_bundle = getattr(self, "_spec_bundle", None)
        if heuristic_policy is None and spec_bundle is not None:
            profile_name = heuristic_public_profile_name_for_policy_id(policy_key)
            if profile_name is not None:
                try:
                    heuristic_policy = HeuristicPublicPolicy.from_spec_bundle(
                        spec_bundle,
                        scoring_profile=profile_name,
                    )
                except Exception:
                    heuristic_policy = None
                else:
                    heuristic_policies[policy_key] = heuristic_policy
        return heuristic_policy

    def _heuristic_public_actions_from_ids(
        self: Any,
        *,
        actor: Any | None,
        heuristic_policy: HeuristicPublicPolicy,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None = None,
        counters: dict[str, int] | None = None,
        profile_name: str | None = None,
    ) -> np.ndarray:
        actions = np.zeros((int(row_indices.shape[0]),), dtype=np.int64)
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        if counters is not None:
            counters["tactical_row_count"] += int(row_indices_array.shape[0])
            counters["fixed_opponent_tactical_row_count"] += int(row_indices_array.shape[0])
        if self._simulator_native_fixed_opponent_available(actor):
            native_actions = self._heuristic_public_actions_from_pool(
                actor=actor,
                row_indices=row_indices_array,
                profile_name=profile_name,
            )
            if native_actions is not None:
                if counters is not None:
                    candidate_counts = np.maximum(
                        np.asarray(legal_offsets[row_indices_array + 1], dtype=np.int64)
                        - np.asarray(legal_offsets[row_indices_array], dtype=np.int64),
                        0,
                    )
                    counters["packed_candidate_count"] += int(candidate_counts.sum())
                return native_actions
        batch_choose = getattr(heuristic_policy, "choose_actions_from_meta_batch", None)
        if callable(batch_choose):
            if legal_action_meta is None:
                subset_ids, subset_offsets = slice_packed_rows(
                    legal_ids,
                    legal_offsets,
                    row_indices_array,
                )
                subset_meta = None
            else:
                subset_ids, subset_offsets, subset_meta = slice_packed_rows_with_meta(
                    legal_ids,
                    legal_offsets,
                    row_indices_array,
                    legal_action_meta=legal_action_meta,
                )
            if counters is not None:
                counters["packed_candidate_count"] += int(subset_ids.shape[0])
            return np.asarray(
                batch_choose(
                    np.asarray(obs_step[row_indices_array], dtype=np.int32),
                    subset_ids,
                    subset_offsets,
                    subset_meta,
                ),
                dtype=np.int64,
            )
        for offset, row_index in enumerate(row_indices_array):
            start = int(legal_offsets[int(row_index)])
            stop = int(legal_offsets[int(row_index) + 1])
            if counters is not None:
                counters["packed_candidate_count"] += max(0, stop - start)
            actions[offset] = int(
                heuristic_policy.choose_action_from_meta(
                    np.asarray(obs_step[int(row_index)]),
                    np.asarray(legal_ids[start:stop], dtype=np.uint32),
                    (None if legal_action_meta is None else np.asarray(legal_action_meta[start:stop], dtype=np.uint16)),
                )
            )
        return actions

    def _heuristic_public_actions_from_mask(
        self: Any,
        *,
        actor: Any | None,
        heuristic_policy: HeuristicPublicPolicy,
        row_indices: np.ndarray,
        obs_step: np.ndarray,
        legal_mask: np.ndarray,
        counters: dict[str, int] | None = None,
        profile_name: str | None = None,
    ) -> np.ndarray:
        actions = np.zeros((int(row_indices.shape[0]),), dtype=np.int64)
        row_indices_array = np.asarray(row_indices, dtype=np.int64)
        if counters is not None:
            counters["tactical_row_count"] += int(row_indices_array.shape[0])
            counters["fixed_opponent_tactical_row_count"] += int(row_indices_array.shape[0])
        if self._simulator_native_fixed_opponent_available(actor):
            native_actions = self._heuristic_public_actions_from_pool(
                actor=actor,
                row_indices=row_indices_array,
                profile_name=profile_name,
            )
            if native_actions is not None:
                if counters is not None:
                    counters["packed_candidate_count"] += int(
                        np.count_nonzero(np.asarray(legal_mask[row_indices_array], dtype=np.bool_))
                    )
                return native_actions
        for offset, row_index in enumerate(row_indices_array.tolist()):
            legal_ids = np.flatnonzero(np.asarray(legal_mask[int(row_index)], dtype=np.bool_)).astype(
                np.uint32, copy=False
            )
            if counters is not None:
                counters["packed_candidate_count"] += int(legal_ids.shape[0])
            actions[offset] = int(heuristic_policy.choose_action(np.asarray(obs_step[int(row_index)]), legal_ids))
        return actions

    def _heuristic_public_actions_from_pool(
        self: Any,
        *,
        actor: Any | None,
        row_indices: np.ndarray,
        profile_name: str | None = None,
    ) -> np.ndarray | None:
        if actor is None:
            raise RuntimeError("simulator_native fixed-opponent routing requires actor context")
        pool = getattr(getattr(actor, "env", None), "pool", None)
        profile_key = str(profile_name or "base").strip().lower()
        env_indices = np.asarray(row_indices, dtype=np.uint32)
        chosen_actions = np.zeros((int(env_indices.shape[0]),), dtype=np.uint16)
        if profile_key and profile_key != "base":
            return None
        choose_into = getattr(pool, "choose_heuristic_public_actions_into", None)
        if not callable(choose_into):
            raise RuntimeError(
                "training.fixed_opponent_backend=simulator_native requires "
                "pool.choose_heuristic_public_actions_into(...)"
            )
        choose_into(env_indices, chosen_actions)
        return chosen_actions.astype(np.int64, copy=False)
