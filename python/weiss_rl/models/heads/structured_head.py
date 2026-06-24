"""Structured legal-action policy head for the policy/value model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.models.heads.structured_head_blueprint import build_structured_head_blueprint
from weiss_rl.models.heads.structured_head_build_plan import STRUCTURED_HEAD_BUILD_PLAN
from weiss_rl.models.heads.structured_head_context import StructuredHeadContextMixin
from weiss_rl.models.heads.structured_head_dimensions import install_candidate_feature_offsets
from weiss_rl.models.heads.structured_head_modules import (
    install_factorized_query_modules,
    install_structured_representation_modules,
)
from weiss_rl.models.heads.structured_head_scoring_surfaces import STRUCTURED_HEAD_SCORING_SURFACES
from weiss_rl.models.heads.structured_head_setup import (
    install_factorized_action_lookup_tables,
    install_structured_action_catalog_view,
    install_structured_action_component_tables,
    resolve_public_heuristic_actor_scale,
    validate_structured_head_inputs,
)
from weiss_rl.models.observations.observation_contract import StructuredObservationContract
from weiss_rl.models.public_heuristic.public_heuristic_scoring import StructuredPublicHeuristicScoringMixin
from weiss_rl.models.public_heuristic.public_heuristic_slots import public_heuristic_slot_preference_array
from weiss_rl.models.scoring.dense_scoring import StructuredDenseScoringMixin
from weiss_rl.models.scoring.factorized_scoring import StructuredFactorizedScoringMixin
from weiss_rl.models.scoring.packed_scoring import StructuredPackedScoringMixin
from weiss_rl.models.scoring.structured_legal_scoring import StructuredLegalActionScoringMixin

_StructuredObservationContract = StructuredObservationContract


class _StructuredLegalActionHead(
    StructuredLegalActionScoringMixin,
    StructuredDenseScoringMixin,
    StructuredPackedScoringMixin,
    StructuredFactorizedScoringMixin,
    StructuredPublicHeuristicScoringMixin,
    StructuredHeadContextMixin,
    nn.Module,
):
    build_plan = STRUCTURED_HEAD_BUILD_PLAN
    scoring_surfaces = STRUCTURED_HEAD_SCORING_SURFACES

    def __init__(
        self,
        *,
        latent_width: int,
        action_catalog: ActionCatalog,
        observation_contract: _StructuredObservationContract,
        card_table: Mapping[str, Any] | None,
        action_feature_width: int,
        layer_norm: bool,
        dropout_p: float,
        candidate_scoring_chunk_size: int = 65536,
        cuda_learner_candidate_scoring_chunk_size: int = 262144,
        public_heuristic_logit_bias_scale: float = 0.0,
        public_heuristic_actor_logit_bias_scale: float = -1.0,
        public_heuristic_logit_bias_families: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        validate_structured_head_inputs(
            latent_width=latent_width,
            action_feature_width=action_feature_width,
            candidate_scoring_chunk_size=candidate_scoring_chunk_size,
            cuda_learner_candidate_scoring_chunk_size=cuda_learner_candidate_scoring_chunk_size,
            public_heuristic_logit_bias_scale=public_heuristic_logit_bias_scale,
            public_heuristic_actor_logit_bias_scale=public_heuristic_actor_logit_bias_scale,
        )
        self.action_dim = int(action_catalog.action_space_size)
        self._stage_slot_count = max(int(action_catalog.max_stage), 1)
        self._observation_contract = observation_contract
        self._card_vocab_size = 32768
        self._public_heuristic_logit_bias_scale = float(public_heuristic_logit_bias_scale)
        self._public_heuristic_actor_logit_bias_scale = resolve_public_heuristic_actor_scale(
            learner_scale=public_heuristic_logit_bias_scale,
            actor_scale=public_heuristic_actor_logit_bias_scale,
        )

        blueprint = build_structured_head_blueprint(
            action_catalog=action_catalog,
            action_dim=int(self.action_dim),
            action_feature_width=action_feature_width,
            public_heuristic_logit_bias_families=public_heuristic_logit_bias_families,
        )
        install_structured_action_catalog_view(self, blueprint.catalog_view)
        install_candidate_feature_offsets(self, dimensions=blueprint.dimensions, offsets=blueprint.offsets)

        install_structured_representation_modules(
            self,
            latent_width=latent_width,
            action_catalog=action_catalog,
            card_table=card_table,
            dimensions=blueprint.dimensions,
            family_count=blueprint.family_count,
            attack_type_count=len(blueprint.catalog_view.attack_type_names),
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        self._candidate_scoring_chunk_size = int(candidate_scoring_chunk_size)
        self._cuda_learner_candidate_scoring_chunk_size = int(cuda_learner_candidate_scoring_chunk_size)
        install_structured_action_component_tables(self, blueprint.action_tables)
        install_factorized_query_modules(
            self,
            factorized_tables=blueprint.factorized_tables,
            dimensions=blueprint.dimensions,
            family_count=blueprint.family_count,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
        install_factorized_action_lookup_tables(self, blueprint.factorized_tables)
        self.register_buffer(
            "_public_slot_preference",
            torch.as_tensor(public_heuristic_slot_preference_array(self._stage_slot_count), dtype=torch.float32),
            persistent=False,
        )
        self._factorized_learner_row_chunk_size = 8192
        self._factorized_actor_row_chunk_size = 32768

    def set_public_heuristic_logit_bias_scales(
        self,
        *,
        learner_scale: float | None = None,
        actor_scale: float | None = None,
    ) -> None:
        if learner_scale is not None:
            resolved = float(learner_scale)
            if resolved < 0.0:
                raise ValueError(f"public_heuristic_logit_bias_scale must be >= 0.0, got {resolved}")
            self._public_heuristic_logit_bias_scale = resolved
        if actor_scale is not None:
            resolved = float(actor_scale)
            if resolved < 0.0:
                raise ValueError(f"public_heuristic_actor_logit_bias_scale must be >= 0.0, got {resolved}")
            self._public_heuristic_actor_logit_bias_scale = resolved
