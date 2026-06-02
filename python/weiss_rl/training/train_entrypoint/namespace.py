"""Namespace installation for the training entrypoint compatibility facade."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.core_aliases import CORE_NAMESPACE_ALIASES
from weiss_rl.training.train_entrypoint.core_exports import CORE_COMPAT_EXPORTS
from weiss_rl.training.train_entrypoint.eval_aliases import EVAL_NAMESPACE_ALIASES
from weiss_rl.training.train_entrypoint.eval_exports import EVAL_COMPAT_EXPORTS
from weiss_rl.training.train_entrypoint.guard_aliases import CHECKPOINT_GUARD_ALIASES
from weiss_rl.training.train_entrypoint.training_aliases import TRAINING_NAMESPACE_ALIASES
from weiss_rl.training.train_entrypoint.training_exports import TRAINING_COMPAT_EXPORTS

COMPAT_EXPORT_FAMILIES: tuple[Mapping[str, Any], ...] = (
    CORE_COMPAT_EXPORTS,
    TRAINING_COMPAT_EXPORTS,
    EVAL_COMPAT_EXPORTS,
)

NAMESPACE_ALIAS_FAMILIES: tuple[Mapping[str, str], ...] = (
    CORE_NAMESPACE_ALIASES,
    TRAINING_NAMESPACE_ALIASES,
    EVAL_NAMESPACE_ALIASES,
)


def install_train_entrypoint_compat_exports(namespace: MutableMapping[str, Any]) -> None:
    for exports in COMPAT_EXPORT_FAMILIES:
        namespace.update(exports)


def install_train_entrypoint_aliases(
    namespace: MutableMapping[str, Any],
    *,
    checkpoint_guard_helpers: Any,
) -> None:
    for aliases in NAMESPACE_ALIAS_FAMILIES:
        for alias_name, source_name in aliases.items():
            namespace[alias_name] = namespace[source_name]
    for alias_name, helper_name in CHECKPOINT_GUARD_ALIASES.items():
        namespace[alias_name] = getattr(checkpoint_guard_helpers, helper_name)
