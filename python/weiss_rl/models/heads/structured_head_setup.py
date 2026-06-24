"""Setup records for the structured legal-action head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.models.actions.action_tables import FactorizedActionLookupTables, StructuredActionComponentTables
from weiss_rl.models.heads.structured_head_dimensions import (
    StructuredCandidateFeatureOffsets,
    StructuredHeadDimensions,
    install_candidate_feature_offsets,
    resolve_candidate_feature_offsets,
    resolve_structured_head_dimensions,
)


@dataclass(frozen=True)
class StructuredActionFamilyIds:
    attack: int
    encore_pay: int
    encore_decline: int
    play_character: int
    main_event: int
    clock_from_hand: int
    climax_play: int
    mulligan_select: int
    mulligan_confirm: int
    main_move: int
    choice_select: int
    next_page: int
    prev_page: int
    level_up: int
    trigger_order: int
    pass_action: int
    hand_families: tuple[int, ...]


@dataclass(frozen=True)
class StructuredAttackTypeIds:
    frontal: int
    side: int
    direct: int


@dataclass(frozen=True)
class StructuredActionCatalogView:
    family_names: tuple[str, ...]
    family_index: dict[str, int]
    attack_type_names: tuple[str, ...]
    attack_type_index: dict[str, int]
    family_ids: StructuredActionFamilyIds
    attack_type_ids: StructuredAttackTypeIds
    public_bias_family_ids: tuple[int, ...]
    meta_unused: int


def validate_structured_head_inputs(
    *,
    latent_width: int,
    action_feature_width: int,
    candidate_scoring_chunk_size: int,
    cuda_learner_candidate_scoring_chunk_size: int,
    public_heuristic_logit_bias_scale: float,
    public_heuristic_actor_logit_bias_scale: float,
) -> None:
    if latent_width <= 0:
        raise ValueError(f"latent_width must be >= 1, got {latent_width}")
    if action_feature_width <= 0:
        raise ValueError(f"action_feature_width must be >= 1, got {action_feature_width}")
    if candidate_scoring_chunk_size <= 0:
        raise ValueError(f"candidate_scoring_chunk_size must be >= 1, got {candidate_scoring_chunk_size}")
    if cuda_learner_candidate_scoring_chunk_size <= 0:
        raise ValueError(
            f"cuda_learner_candidate_scoring_chunk_size must be >= 1, got {cuda_learner_candidate_scoring_chunk_size}"
        )
    if public_heuristic_logit_bias_scale < 0.0:
        raise ValueError(f"public_heuristic_logit_bias_scale must be >= 0.0, got {public_heuristic_logit_bias_scale}")
    if public_heuristic_actor_logit_bias_scale < 0.0 and public_heuristic_actor_logit_bias_scale != -1.0:
        raise ValueError(
            "public_heuristic_actor_logit_bias_scale must be >= 0.0 or -1.0, "
            f"got {public_heuristic_actor_logit_bias_scale}"
        )


def resolve_public_heuristic_actor_scale(*, learner_scale: float, actor_scale: float) -> float:
    return float(learner_scale if actor_scale < 0.0 else actor_scale)


def resolve_structured_action_catalog_view(
    *,
    action_catalog: ActionCatalog,
    public_heuristic_logit_bias_families: tuple[str, ...],
) -> StructuredActionCatalogView:
    family_names = tuple(family.name for family in action_catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    unknown_public_bias_families = sorted(
        {name for name in public_heuristic_logit_bias_families if name not in family_index}
    )
    if unknown_public_bias_families:
        raise ValueError(
            "public_heuristic_logit_bias_families contains unknown action families: "
            + ", ".join(unknown_public_bias_families)
        )

    attack_type_names = tuple(action_catalog.attack_type_names)
    attack_type_index = {name: index for index, name in enumerate(attack_type_names)}
    main_event = int(family_index.get("main_play_event", -1))
    clock_from_hand = int(family_index.get("clock_from_hand", -1))
    climax_play = int(family_index.get("climax_play", -1))
    mulligan_select = int(family_index.get("mulligan_select", -1))
    family_ids = StructuredActionFamilyIds(
        attack=int(family_index.get("attack", -1)),
        encore_pay=int(family_index.get("encore_pay", -1)),
        encore_decline=int(family_index.get("encore_decline", -1)),
        play_character=int(family_index.get("main_play_character", -1)),
        main_event=main_event,
        clock_from_hand=clock_from_hand,
        climax_play=climax_play,
        mulligan_select=mulligan_select,
        mulligan_confirm=int(family_index.get("mulligan_confirm", -1)),
        main_move=int(family_index.get("main_move", -1)),
        choice_select=int(family_index.get("choice_select", -1)),
        next_page=int(family_index.get("choice_next_page", -1)),
        prev_page=int(family_index.get("choice_prev_page", -1)),
        level_up=int(family_index.get("level_up", -1)),
        trigger_order=int(family_index.get("trigger_order", -1)),
        pass_action=int(family_index.get("pass", -1)),
        hand_families=tuple(
            family_id
            for family_id in (
                main_event,
                clock_from_hand,
                climax_play,
                mulligan_select,
            )
            if family_id >= 0
        ),
    )
    attack_type_ids = StructuredAttackTypeIds(
        frontal=int(attack_type_index.get("frontal", -1)),
        side=int(attack_type_index.get("side", -1)),
        direct=int(attack_type_index.get("direct", -1)),
    )
    return StructuredActionCatalogView(
        family_names=family_names,
        family_index=family_index,
        attack_type_names=attack_type_names,
        attack_type_index=attack_type_index,
        family_ids=family_ids,
        attack_type_ids=attack_type_ids,
        public_bias_family_ids=tuple(int(family_index[name]) for name in public_heuristic_logit_bias_families),
        meta_unused=int(np.iinfo(np.uint16).max),
    )


def install_structured_action_catalog_view(
    head: Any,
    catalog_view: StructuredActionCatalogView,
) -> None:
    family_ids = catalog_view.family_ids
    attack_type_ids = catalog_view.attack_type_ids
    head._meta_unused = catalog_view.meta_unused
    head._attack_family_id = family_ids.attack
    head._encore_pay_family_id = family_ids.encore_pay
    head._encore_decline_family_id = family_ids.encore_decline
    head._play_character_family_id = family_ids.play_character
    head._main_event_family_id = family_ids.main_event
    head._clock_from_hand_family_id = family_ids.clock_from_hand
    head._climax_play_family_id = family_ids.climax_play
    head._mulligan_select_family_id = family_ids.mulligan_select
    head._mulligan_confirm_family_id = family_ids.mulligan_confirm
    head._main_move_family_id = family_ids.main_move
    head._choice_select_family_id = family_ids.choice_select
    head.register_buffer(
        "_public_heuristic_bias_family_ids",
        torch.as_tensor(catalog_view.public_bias_family_ids, dtype=torch.long),
        persistent=False,
    )
    head._next_page_family_id = family_ids.next_page
    head._prev_page_family_id = family_ids.prev_page
    head._level_up_family_id = family_ids.level_up
    head._trigger_order_family_id = family_ids.trigger_order
    head._pass_family_id = family_ids.pass_action
    head._frontal_attack_type_id = attack_type_ids.frontal
    head._side_attack_type_id = attack_type_ids.side
    head._direct_attack_type_id = attack_type_ids.direct
    head._hand_family_ids = family_ids.hand_families


def install_structured_action_component_tables(
    head: Any,
    action_tables: StructuredActionComponentTables,
) -> None:
    head.register_buffer("_family_ids", torch.as_tensor(action_tables.family_ids, dtype=torch.long))
    head.register_buffer("_action_arg0", torch.as_tensor(action_tables.action_arg0, dtype=torch.long))
    head.register_buffer("_action_arg1", torch.as_tensor(action_tables.action_arg1, dtype=torch.long))
    head.register_buffer("_hand_indices", torch.as_tensor(action_tables.hand_indices, dtype=torch.long))
    head.register_buffer("_stage_slots", torch.as_tensor(action_tables.stage_slots, dtype=torch.long))
    head.register_buffer("_from_slots", torch.as_tensor(action_tables.from_slots, dtype=torch.long))
    head.register_buffer("_to_slots", torch.as_tensor(action_tables.to_slots, dtype=torch.long))
    head.register_buffer("_attack_slots", torch.as_tensor(action_tables.attack_slots, dtype=torch.long))
    head.register_buffer("_attack_types", torch.as_tensor(action_tables.attack_types, dtype=torch.long))
    head.register_buffer("_generic_indices", torch.as_tensor(action_tables.generic_indices, dtype=torch.long))


def install_factorized_action_lookup_tables(
    head: Any,
    factorized_tables: FactorizedActionLookupTables,
) -> None:
    head.register_buffer("_family_arg_kind", torch.as_tensor(factorized_tables.family_arg_kind, dtype=torch.long))
    head.register_buffer(
        "_family_arg0_size",
        torch.as_tensor(factorized_tables.family_arg0_size, dtype=torch.long),
    )
    head.register_buffer(
        "_family_arg1_size",
        torch.as_tensor(factorized_tables.family_arg1_size, dtype=torch.long),
    )
    head.register_buffer(
        "_family_noarg_action_ids",
        torch.as_tensor(factorized_tables.family_noarg_action_ids, dtype=torch.long),
    )
    head.register_buffer(
        "_one_arg_action_ids",
        torch.as_tensor(factorized_tables.one_arg_action_ids, dtype=torch.long),
    )
    head.register_buffer(
        "_two_arg_action_ids",
        torch.as_tensor(factorized_tables.two_arg_action_ids, dtype=torch.long),
    )
    head._slot_family_ids = factorized_tables.slot_family_ids
    head._index_family_ids = factorized_tables.index_family_ids


__all__ = [
    "StructuredActionCatalogView",
    "StructuredActionFamilyIds",
    "StructuredAttackTypeIds",
    "StructuredCandidateFeatureOffsets",
    "StructuredHeadDimensions",
    "install_candidate_feature_offsets",
    "install_factorized_action_lookup_tables",
    "install_structured_action_component_tables",
    "install_structured_action_catalog_view",
    "resolve_candidate_feature_offsets",
    "resolve_public_heuristic_actor_scale",
    "resolve_structured_action_catalog_view",
    "resolve_structured_head_dimensions",
    "validate_structured_head_inputs",
]
