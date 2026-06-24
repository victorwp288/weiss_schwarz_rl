"""Deterministic public-only heuristic policies used for heuristic anchors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog as _ActionCatalog
from weiss_rl.core.action_catalog import DecodedAction as _DecodedAction
from weiss_rl.eval.heuristic_public.heuristic_public_action_scoring import (
    HeuristicPublicActionScorer as _HeuristicPublicActionScorer,
)
from weiss_rl.eval.heuristic_public.heuristic_public_batch_selection import (
    HeuristicPublicMetaBatchSelector as _HeuristicPublicMetaBatchSelector,
)
from weiss_rl.eval.heuristic_public.heuristic_public_observation import (
    PublicObservationLayout as _PublicObservationLayout,
)
from weiss_rl.public_heuristic.profiles import (
    HeuristicPublicScoringProfile as _HeuristicPublicScoringProfile,
)
from weiss_rl.public_heuristic.profiles import (
    heuristic_public_scoring_profile as _heuristic_public_scoring_profile,
)


class HeuristicPublicPolicy:
    """Deterministic action selection that only consults public observation features."""

    def __init__(
        self,
        *,
        action_catalog: _ActionCatalog,
        observation_layout: _PublicObservationLayout,
        scoring_profile: _HeuristicPublicScoringProfile | str = "base",
    ) -> None:
        self._action_catalog = action_catalog
        self._observation_layout = observation_layout
        self._scoring_profile = (
            scoring_profile
            if isinstance(scoring_profile, _HeuristicPublicScoringProfile)
            else _heuristic_public_scoring_profile(str(scoring_profile))
        )
        self._action_scorer = _HeuristicPublicActionScorer(self._scoring_profile)
        self._decode_cache: dict[int, _DecodedAction] = {}
        self._family_index = {family.name: index for index, family in enumerate(self._action_catalog.families)}
        self._attack_type_index = {
            str(name): index for index, name in enumerate(self._action_catalog.attack_type_names)
        }
        self._meta_batch_selector = _HeuristicPublicMetaBatchSelector(
            observation_layout=self._observation_layout,
            scoring_profile=self._scoring_profile,
            pass_action_id=self.pass_action_id,
            family_index=self._family_index,
            attack_type_index=self._attack_type_index,
        )

    @classmethod
    def from_spec_bundle(
        cls,
        spec_bundle: Mapping[str, object],
        *,
        scoring_profile: _HeuristicPublicScoringProfile | str = "base",
    ) -> HeuristicPublicPolicy:
        action_catalog = _ActionCatalog.from_spec_bundle(spec_bundle)
        observation_layout = _PublicObservationLayout.from_spec_bundle(
            spec_bundle,
            stage_slot_count=action_catalog.max_stage,
        )
        return cls(
            action_catalog=action_catalog,
            observation_layout=observation_layout,
            scoring_profile=scoring_profile,
        )

    @property
    def pass_action_id(self) -> int:
        return self._action_catalog.pass_action_id

    def choose_action(self, obs_row: np.ndarray, legal_ids: np.ndarray) -> int:
        if np.asarray(legal_ids).size == 0:
            return self.pass_action_id
        board = self._observation_layout.parse_public_board(obs_row)
        best_action_id = self.pass_action_id
        best_score: tuple[int, ...] | None = None
        for action_id in np.asarray(legal_ids, dtype=np.int64).tolist():
            decoded = self._decode(int(action_id))
            candidate_score = self._action_scorer.score_action(decoded, board) + (-int(action_id),)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_action_id = int(action_id)
        return best_action_id

    def choose_action_from_meta(
        self,
        obs_row: np.ndarray,
        legal_ids: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> int:
        action_ids = np.asarray(legal_ids, dtype=np.int64).reshape(-1)
        if action_ids.size == 0:
            return self.pass_action_id
        if legal_action_meta is None:
            return self.choose_action(obs_row, action_ids.astype(np.uint32, copy=False))
        meta = np.asarray(legal_action_meta, dtype=np.uint16)
        if meta.ndim != 2 or meta.shape[0] != action_ids.shape[0] or meta.shape[1] < 3:
            return self.choose_action(obs_row, action_ids.astype(np.uint32, copy=False))
        return int(
            self.choose_actions_from_meta_batch(
                np.asarray(obs_row, dtype=np.int32).reshape(1, -1),
                action_ids.astype(np.uint32, copy=False),
                np.asarray([0, action_ids.shape[0]], dtype=np.uint32),
                meta,
            )[0]
        )

    def choose_actions_from_meta_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray:
        obs_batch = np.asarray(obs_rows, dtype=np.int32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if obs_batch.ndim != 2:
            raise ValueError("obs_rows must have shape (rows, observation)")
        if obs_batch.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        if obs_batch.shape[1] < self._observation_layout.obs_len:
            raise ValueError(
                f"observation rows are too short ({obs_batch.shape[1]} < {self._observation_layout.obs_len})"
            )
        action_ids = np.asarray(legal_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(legal_offsets, dtype=np.int64).reshape(-1)
        if offsets.shape != (obs_batch.shape[0] + 1,):
            return self._choose_actions_scalar_batch(obs_batch, action_ids)
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(action_ids.shape[0]) or np.any(np.diff(offsets) < 0):
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)
        selected_actions = self._meta_batch_selector.choose_actions(
            obs_batch=obs_batch,
            action_ids=action_ids,
            offsets=offsets,
            legal_action_meta=legal_action_meta,
        )
        if selected_actions is None:
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)
        return selected_actions

    def _choose_actions_scalar_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        *,
        offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        obs_batch = np.asarray(obs_rows, dtype=np.int32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        action_ids = np.asarray(legal_ids, dtype=np.uint32).reshape(-1)
        if offsets is None:
            offsets = np.asarray([0, action_ids.shape[0]], dtype=np.int64)
        chosen_actions = np.full((obs_batch.shape[0],), self.pass_action_id, dtype=np.int64)
        for row_index in range(obs_batch.shape[0]):
            start = int(offsets[row_index])
            stop = int(offsets[row_index + 1])
            chosen_actions[row_index] = int(self.choose_action(obs_batch[row_index], action_ids[start:stop]))
        return chosen_actions

    def _decode(self, action_id: int) -> _DecodedAction:
        cached = self._decode_cache.get(int(action_id))
        if cached is not None:
            return cached
        decoded = self._action_catalog.decode(int(action_id))
        self._decode_cache[int(action_id)] = decoded
        return decoded


__all__ = [
    "HeuristicPublicPolicy",
]
