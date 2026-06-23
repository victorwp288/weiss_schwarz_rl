"""Action-surface guards applied before eval model scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.models.observation_contract import header_field_index
from weiss_rl.runtime.components.action_surface import (
    filter_batch_main_move_only_rows_to_pass,
    filter_batch_mulligan_select_after_select,
    filter_batch_pass_when_attack_available,
)
from weiss_rl.runtime.components.legal_meta import action_catalog_indices


@dataclass(frozen=True, slots=True)
class ModelActionSurfaceSettings:
    pass_action_id: int
    mulligan_force_confirm_after_select: bool = False
    force_pass_over_main_move_only: bool = False
    main_move_only_max_consecutive: int = 0
    force_attack_over_pass_when_attack_legal: bool = False

    @classmethod
    def from_training_config(cls, training_config: Any, *, pass_action_id: int) -> ModelActionSurfaceSettings:
        return cls(
            pass_action_id=int(pass_action_id),
            mulligan_force_confirm_after_select=bool(
                getattr(training_config, "mulligan_force_confirm_after_select", False)
            ),
            force_pass_over_main_move_only=bool(getattr(training_config, "force_pass_over_main_move_only", False)),
            main_move_only_max_consecutive=int(getattr(training_config, "main_move_only_max_consecutive", 0)),
            force_attack_over_pass_when_attack_legal=bool(
                getattr(training_config, "force_attack_over_pass_when_attack_legal", False)
            ),
        )

    @property
    def has_guards(self) -> bool:
        return (
            self.mulligan_force_confirm_after_select
            or self.force_pass_over_main_move_only
            or self.force_attack_over_pass_when_attack_legal
        )


def model_action_surface_batch_and_ids(
    *,
    model: Any | None,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    settings: ModelActionSurfaceSettings,
    action_sequence_state: Any | None = None,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    if not settings.has_guards or model is None:
        return batch, legal_ids
    action_catalog = getattr(model, "action_catalog", None)
    if action_catalog is None:
        return batch, legal_ids
    contract = getattr(model, "_structured_observation_contract", None)
    return action_catalog_action_surface_batch_and_ids(
        action_catalog=action_catalog,
        observation_layout=getattr(contract, "layout", None),
        batch=batch,
        legal_ids=legal_ids,
        settings=settings,
        action_sequence_state=action_sequence_state,
    )


def action_catalog_action_surface_batch_and_ids(
    *,
    action_catalog: ActionCatalog,
    observation_layout: Any | None,
    batch: DecisionBoundaryBatch,
    legal_ids: np.ndarray,
    settings: ModelActionSurfaceSettings,
    action_sequence_state: Any | None = None,
) -> tuple[DecisionBoundaryBatch, np.ndarray]:
    if not settings.has_guards:
        return batch, legal_ids

    filtered_batch = batch
    family_index, _attack_type_index = action_catalog_indices(action_catalog)
    if settings.mulligan_force_confirm_after_select:
        filtered_batch, _result = filter_batch_mulligan_select_after_select(
            filtered_batch,
            last_action_arg0_index=_last_action_arg0_index(observation_layout),
            mulligan_select_family_id=int(family_index.get("mulligan_select", -1)),
            mulligan_confirm_family_id=int(family_index.get("mulligan_confirm", -1)),
        )
    if settings.force_pass_over_main_move_only:
        filtered_batch, _result = filter_batch_main_move_only_rows_to_pass(
            filtered_batch,
            pass_action_id=int(settings.pass_action_id),
            main_move_family_id=int(family_index.get("main_move", -1)),
            allow_main_move_only_rows=_allow_main_move_only_rows(settings, action_sequence_state),
        )
    if settings.force_attack_over_pass_when_attack_legal:
        filtered_batch, _result = filter_batch_pass_when_attack_available(
            filtered_batch,
            pass_action_id=int(settings.pass_action_id),
            attack_family_id=int(family_index.get("attack", -1)),
        )
    if filtered_batch.ids_offsets is None:
        return batch, legal_ids
    filtered_ids, filtered_offsets = filtered_batch.ids_offsets
    return (
        filtered_batch,
        np.asarray(filtered_ids[int(filtered_offsets[0]) : int(filtered_offsets[1])], dtype=np.uint32),
    )


def _last_action_arg0_index(observation_layout: Any | None) -> int:
    field_index = None if observation_layout is None else header_field_index(observation_layout, "last_action_arg0")
    return -1 if field_index is None else int(field_index)


def _allow_main_move_only_rows(
    settings: ModelActionSurfaceSettings,
    action_sequence_state: Any | None,
) -> np.ndarray | None:
    if settings.main_move_only_max_consecutive <= 0 or action_sequence_state is None:
        return None
    consecutive = np.asarray(action_sequence_state.consecutive_main_moves_by_env, dtype=np.int32)
    if consecutive.shape != (1,):
        return None
    return consecutive < int(settings.main_move_only_max_consecutive)
