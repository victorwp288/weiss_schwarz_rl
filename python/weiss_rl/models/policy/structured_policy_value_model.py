"""Structured policy/value model implementation.

This is the thesis model path. It keeps the dense model trunk/value behavior,
then replaces the flat policy head with a simulator-aware legal-candidate head.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch

from weiss_rl.config.models import ModelConfig
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.card_table import cached_runtime_card_table
from weiss_rl.models.backbone.base import GLOBAL_ACTION_SPACE_SIZE, STRUCTURED_V2_ENCODER_KIND
from weiss_rl.models.heads.structured_head import _StructuredLegalActionHead
from weiss_rl.models.observations.observation_contract import build_structured_observation_contract
from weiss_rl.models.policy.policy_value_facade import StructuredLegalPolicyValueFacadeMixin
from weiss_rl.models.policy.policy_value_model import PolicyValueModel


class StructuredLegalPolicyValueModel(StructuredLegalPolicyValueFacadeMixin, PolicyValueModel):
    """Actor-critic that scores simulator-provided legal candidates."""

    def __init__(
        self,
        *,
        observation_dim: int,
        config: ModelConfig,
        action_dim: int = GLOBAL_ACTION_SPACE_SIZE,
        dropout_p: float | None = None,
        observation_spec: Mapping[str, Any] | None = None,
        spec_bundle: Mapping[str, Any] | None = None,
        card_table: Mapping[str, Any] | None = None,
    ) -> None:
        if spec_bundle is None:
            raise ValueError("structured_v2 encoder requires the simulator spec bundle")
        action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
        observation_contract = build_structured_observation_contract(
            spec_bundle["observation"],
            action_catalog=action_catalog,
        )
        structured_config = replace(config, encoder_kind="typed_v1")
        super().__init__(
            observation_dim=observation_dim,
            config=structured_config,
            action_dim=action_dim,
            dropout_p=dropout_p,
            observation_spec=observation_spec,
        )
        if action_catalog.action_space_size != action_dim:
            raise ValueError(
                "structured_v2 action catalog mismatch: "
                f"expected {action_dim}, observed {action_catalog.action_space_size}"
            )
        encoder_dropout = structured_config.dropout.family_a if dropout_p is None else dropout_p
        action_feature_width = max(32, int(structured_config.encoder_mlp_width))
        self.policy_head = cast(
            Any,
            _StructuredLegalActionHead(
                latent_width=int(structured_config.gru_hidden_size),
                action_catalog=action_catalog,
                observation_contract=observation_contract,
                card_table=cached_runtime_card_table() if card_table is None else card_table,
                action_feature_width=action_feature_width,
                layer_norm=bool(structured_config.layer_norm),
                dropout_p=float(encoder_dropout),
                candidate_scoring_chunk_size=int(structured_config.candidate_scoring_chunk_size),
                cuda_learner_candidate_scoring_chunk_size=int(
                    structured_config.cuda_learner_candidate_scoring_chunk_size
                ),
                public_heuristic_logit_bias_scale=float(structured_config.public_heuristic_logit_bias_scale),
                public_heuristic_actor_logit_bias_scale=float(
                    structured_config.public_heuristic_actor_logit_bias_scale
                ),
                public_heuristic_logit_bias_families=tuple(structured_config.public_heuristic_logit_bias_families),
            ),
        )
        self.action_catalog = action_catalog
        self._structured_observation_contract = observation_contract
        self.register_buffer(
            "_card_scalar_indices",
            torch.as_tensor(observation_contract.card_scalar_indices, dtype=torch.long),
            persistent=False,
        )
        encoder_keep_mask = torch.ones((int(observation_dim),), dtype=torch.float32)
        if observation_contract.card_scalar_indices:
            encoder_keep_mask[torch.as_tensor(observation_contract.card_scalar_indices, dtype=torch.long)] = 0.0
        self.register_buffer("_encoder_input_keep_mask", encoder_keep_mask, persistent=False)
        self.supports_legal_candidate_scoring = True
        self.structured_policy_contract = str(config.structured_policy_contract).strip().lower()
        self.supports_factorized_legal_policy = self.structured_policy_contract == "factorized_v1"
        self.encoder_kind = STRUCTURED_V2_ENCODER_KIND
        self._compiled_trunk_packed_core: Any | None = None
        self._compiled_trunk_sequence_core: Any | None = None
        self._trunk_compile_last_error: str | None = None
        self._install_candidate_residual_adapter(config=config)


StructuredLegalPolicyValueModel.__module__ = "weiss_rl.model"

__all__ = ["StructuredLegalPolicyValueModel"]
