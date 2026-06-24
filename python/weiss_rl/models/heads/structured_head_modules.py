from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.card_table import card_feature_table
from weiss_rl.models.actions.action_tables import FactorizedActionLookupTables
from weiss_rl.models.backbone.layers import build_mlp_stack
from weiss_rl.models.heads.structured_head_dimensions import StructuredHeadDimensions


def install_structured_representation_modules(
    head: Any,
    *,
    latent_width: int,
    action_catalog: ActionCatalog,
    card_table: Mapping[str, Any] | None,
    dimensions: StructuredHeadDimensions,
    family_count: int,
    attack_type_count: int,
    layer_norm: bool,
    dropout_p: float,
) -> None:
    head.family_embedding = nn.Embedding(max(family_count, 1), dimensions.family_embed_dim)
    head.slot_embedding = nn.Embedding(head._stage_slot_count + 1, dimensions.slot_embed_dim)
    head.attack_type_embedding = nn.Embedding(attack_type_count + 1, dimensions.slot_embed_dim)
    head.card_embedding = nn.Embedding(head._card_vocab_size, dimensions.card_embed_dim)
    head.hand_position_embedding = nn.Embedding(
        max(int(action_catalog.max_hand), 1) + 1,
        dimensions.card_embed_dim,
    )
    static_feature_table = card_feature_table(card_table=card_table, vocab_size=head._card_vocab_size)
    head.register_buffer(
        "_card_static_features",
        torch.as_tensor(static_feature_table, dtype=torch.float32),
        persistent=False,
    )
    head.card_feature_projection = (
        None
        if static_feature_table.shape[1] == 0
        else build_mlp_stack(
            input_dim=int(static_feature_table.shape[1]),
            width=dimensions.card_embed_dim,
            layers=1,
            layer_norm=layer_norm,
            dropout_p=dropout_p,
        )
    )
    head.hand_summary_projection = build_mlp_stack(
        input_dim=dimensions.card_embed_dim * 2 + 1,
        width=dimensions.slot_context_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.slot_encoder = build_mlp_stack(
        input_dim=dimensions.card_embed_dim + 7,
        width=dimensions.slot_context_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.state_projection = build_mlp_stack(
        input_dim=latent_width + dimensions.slot_context_dim * 3,
        width=dimensions.state_width,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.candidate_projection = build_mlp_stack(
        input_dim=dimensions.candidate_input_dim,
        width=dimensions.state_width,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    scorer_layers: list[nn.Module] = [nn.Linear(dimensions.state_width * 2, dimensions.state_width)]
    if layer_norm:
        scorer_layers.append(nn.LayerNorm(dimensions.state_width))
    scorer_layers.append(nn.ReLU())
    if dropout_p > 0.0:
        scorer_layers.append(nn.Dropout(p=dropout_p))
    final_scorer = nn.Linear(dimensions.state_width, 1)
    nn.init.zeros_(final_scorer.weight)
    nn.init.zeros_(final_scorer.bias)
    scorer_layers.append(final_scorer)
    head.joint_scorer = nn.Sequential(*scorer_layers)
    head.family_bias = nn.Parameter(torch.zeros(max(family_count, 1)))


def install_factorized_query_modules(
    head: Any,
    *,
    factorized_tables: FactorizedActionLookupTables,
    dimensions: StructuredHeadDimensions,
    family_count: int,
    layer_norm: bool,
    dropout_p: float,
) -> None:
    head.generic_index_embedding = nn.Embedding(
        int(factorized_tables.max_arg0) + 1,
        dimensions.generic_embed_dim,
    )
    head.generic_candidate_projection = build_mlp_stack(
        input_dim=dimensions.generic_embed_dim,
        width=dimensions.card_embed_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.family_head = nn.Linear(dimensions.state_width, family_count)
    nn.init.zeros_(head.family_head.weight)
    nn.init.zeros_(head.family_head.bias)
    head.hand_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim,
        width=dimensions.card_embed_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.index_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim,
        width=dimensions.generic_embed_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.slot_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim,
        width=dimensions.slot_context_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.play_slot_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim + dimensions.card_embed_dim,
        width=dimensions.slot_context_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.move_target_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim + dimensions.slot_context_dim,
        width=dimensions.slot_context_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
    head.attack_type_query_head = build_mlp_stack(
        input_dim=dimensions.state_width + dimensions.family_embed_dim + dimensions.slot_context_dim,
        width=dimensions.slot_embed_dim,
        layers=1,
        layer_norm=layer_norm,
        dropout_p=dropout_p,
    )
