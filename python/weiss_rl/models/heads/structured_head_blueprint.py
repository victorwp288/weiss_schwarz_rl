"""Resolved lookup tables and dimensions for the structured action head."""

from __future__ import annotations

from dataclasses import dataclass

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.models.actions.action_tables import (
    FactorizedActionLookupTables,
    StructuredActionComponentTables,
    build_factorized_action_lookup_tables,
    build_structured_action_component_tables,
)
from weiss_rl.models.heads.structured_head_dimensions import (
    StructuredCandidateFeatureOffsets,
    StructuredHeadDimensions,
    resolve_candidate_feature_offsets,
    resolve_structured_head_dimensions,
)
from weiss_rl.models.heads.structured_head_setup import (
    StructuredActionCatalogView,
    resolve_structured_action_catalog_view,
)


@dataclass(frozen=True, slots=True)
class StructuredHeadBlueprint:
    catalog_view: StructuredActionCatalogView
    action_tables: StructuredActionComponentTables
    dimensions: StructuredHeadDimensions
    offsets: StructuredCandidateFeatureOffsets
    factorized_tables: FactorizedActionLookupTables

    @property
    def family_count(self) -> int:
        return max(len(self.catalog_view.family_names), 1)


def build_structured_head_blueprint(
    *,
    action_catalog: ActionCatalog,
    action_dim: int,
    action_feature_width: int,
    public_heuristic_logit_bias_families: tuple[str, ...],
) -> StructuredHeadBlueprint:
    """Resolve the catalog-dependent tables used by all structured scoring paths."""

    catalog_view = resolve_structured_action_catalog_view(
        action_catalog=action_catalog,
        public_heuristic_logit_bias_families=public_heuristic_logit_bias_families,
    )
    action_tables = build_structured_action_component_tables(
        action_catalog=action_catalog,
        action_dim=int(action_dim),
        family_index=catalog_view.family_index,
        attack_type_index=catalog_view.attack_type_index,
    )
    dimensions = resolve_structured_head_dimensions(action_feature_width)
    offsets = resolve_candidate_feature_offsets(dimensions)
    factorized_tables = build_factorized_action_lookup_tables(
        action_dim=int(action_dim),
        family_count=max(len(catalog_view.family_names), 1),
        family_index=catalog_view.family_index,
        component_tables=action_tables,
    )
    return StructuredHeadBlueprint(
        catalog_view=catalog_view,
        action_tables=action_tables,
        dimensions=dimensions,
        offsets=offsets,
        factorized_tables=factorized_tables,
    )


__all__ = ["StructuredHeadBlueprint", "build_structured_head_blueprint"]
