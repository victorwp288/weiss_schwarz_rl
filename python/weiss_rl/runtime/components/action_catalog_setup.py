"""Runtime action-catalog setup derived from the simulator contract."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.runtime.components.legal_meta import action_catalog_indices


@dataclass(frozen=True)
class RuntimeActionCatalogSetup:
    action_catalog: ActionCatalog | None
    action_family_index: dict[str, int]
    action_attack_type_index: dict[str, int]
    last_action_arg0_obs_index: int


def resolve_runtime_action_catalog_setup(*, spec_bundle: dict[str, Any] | None) -> RuntimeActionCatalogSetup:
    action_catalog: ActionCatalog | None = None
    action_family_index: dict[str, int] = {}
    action_attack_type_index: dict[str, int] = {}
    if spec_bundle is not None:
        with suppress(Exception):
            action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
    if action_catalog is not None:
        action_family_index, action_attack_type_index = action_catalog_indices(action_catalog)

    return RuntimeActionCatalogSetup(
        action_catalog=action_catalog,
        action_family_index=action_family_index,
        action_attack_type_index=action_attack_type_index,
        last_action_arg0_obs_index=_last_action_arg0_obs_index(spec_bundle),
    )


def _last_action_arg0_obs_index(spec_bundle: dict[str, Any] | None) -> int:
    if spec_bundle is None:
        return -1
    observation_spec = spec_bundle.get("observation", {})
    if not isinstance(observation_spec, dict):
        return -1
    for field in observation_spec.get("header_fields", []):
        if isinstance(field, dict) and field.get("name") == "last_action_arg0":
            return int(field.get("index", -1))
    return -1


__all__ = ["RuntimeActionCatalogSetup", "resolve_runtime_action_catalog_setup"]
