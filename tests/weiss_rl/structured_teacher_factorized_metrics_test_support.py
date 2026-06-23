from __future__ import annotations

from collections.abc import Callable

import torch
from weiss_rl.core.action_catalog import ActionCatalog


def family_indices(action_catalog: ActionCatalog) -> dict[str, int]:
    return {family.name: index for index, family in enumerate(action_catalog.families)}


def first_action_id(
    action_catalog: ActionCatalog,
    *,
    family: str,
    predicate: Callable[[object], bool] | None = None,
) -> int:
    return next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == family
        and (predicate is None or predicate(action_catalog.decode(action_id)))
    )


def confident_family_log_probs(action_catalog: ActionCatalog, families: list[str]) -> torch.Tensor:
    indices = family_indices(action_catalog)
    logits = torch.full((len(families), 1, len(action_catalog.families)), -2.0)
    for row, family in enumerate(families):
        logits[row, 0, indices[family]] = 3.0
    return torch.log_softmax(logits, dim=-1)
