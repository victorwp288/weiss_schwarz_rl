from __future__ import annotations

from typing import Any


def _explicit_policy_selection(policy_ids: list[str]) -> tuple[list[str], dict[str, Any]] | None:
    explicit_policy_ids = [policy_id.strip() for policy_id in policy_ids if policy_id.strip()]
    if not explicit_policy_ids:
        return None
    return explicit_policy_ids, {"mode": "explicit_cli", "policy_count": len(explicit_policy_ids)}


def _manifest_policy_selection_fallback(manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]] | None:
    manifest_policy_ids = manifest.get("policy_set_selection")
    resolved_from_manifest = (
        [str(policy_id).strip() for policy_id in manifest_policy_ids if str(policy_id).strip()]
        if isinstance(manifest_policy_ids, list)
        else []
    )
    if not resolved_from_manifest:
        return None
    return (
        resolved_from_manifest,
        {
            "mode": "manifest_policy_set_selection_fallback",
            "policy_count": len(resolved_from_manifest),
        },
    )


__all__ = ["_explicit_policy_selection", "_manifest_policy_selection_fallback"]
