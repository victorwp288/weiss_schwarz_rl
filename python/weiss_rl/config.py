"""Config loading utilities for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import (
    HARD_FAIL_SPEC_MISMATCH_POLICY,
    normalize_bool_flag,
    normalize_spec_mismatch_policy,
    require_fail_on_spec_mismatch,
)


@dataclass(slots=True)
class StackConfig:
    """Top-level pointer map loaded from `configs/rl_stack_locked.yaml`."""

    root: Path
    components: dict[str, Path]
    seed_sets: dict[str, Path]
    spec_mismatch_policy: str = HARD_FAIL_SPEC_MISMATCH_POLICY
    require_export_spec_bundle: bool = False
    persist_spec_bundle_in_manifest: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _mapping_field(parent: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at {source}.{key}")
    return value


def _load_component_contract(components: dict[str, Path]) -> tuple[str, bool, bool]:
    policy = HARD_FAIL_SPEC_MISMATCH_POLICY
    require_export_spec_bundle = False
    persist_spec_bundle_in_manifest = False

    reproducibility_path = components.get("reproducibility")
    if reproducibility_path is not None:
        doc = _load_yaml(reproducibility_path)
        body = doc.get("reproducibility", doc)
        if not isinstance(body, dict):
            raise ValueError(f"Missing `reproducibility` mapping in {reproducibility_path}")

        spec_bundle = _mapping_field(body, "spec_bundle", source=str(reproducibility_path))
        require_export_spec_bundle = normalize_bool_flag(
            spec_bundle.get("require_export_spec_bundle"),
            source=f"{reproducibility_path}: reproducibility.spec_bundle.require_export_spec_bundle",
            default=False,
        )
        persist_spec_bundle_in_manifest = normalize_bool_flag(
            spec_bundle.get("persist_in_manifest"),
            source=f"{reproducibility_path}: reproducibility.spec_bundle.persist_in_manifest",
            default=False,
        )
        policy = require_fail_on_spec_mismatch(
            spec_bundle.get("fail_on_spec_mismatch", True),
            source=f"{reproducibility_path}: reproducibility.spec_bundle.fail_on_spec_mismatch",
        )

        legal_fingerprint = _mapping_field(body, "legal_fingerprint", source=str(reproducibility_path))
        normalize_spec_mismatch_policy(
            legal_fingerprint.get("replay_eval_mismatch_policy"),
            source=f"{reproducibility_path}: reproducibility.legal_fingerprint.replay_eval_mismatch_policy",
        )

    evaluation_path = components.get("evaluation")
    if evaluation_path is not None:
        doc = _load_yaml(evaluation_path)
        body = doc.get("evaluation", doc)
        if not isinstance(body, dict):
            raise ValueError(f"Missing `evaluation` mapping in {evaluation_path}")

        legal_fingerprint_checks = _mapping_field(body, "legal_fingerprint_checks", source=str(evaluation_path))
        normalize_spec_mismatch_policy(
            legal_fingerprint_checks.get("mismatch_policy"),
            source=f"{evaluation_path}: evaluation.legal_fingerprint_checks.mismatch_policy",
        )

    return policy, require_export_spec_bundle, persist_spec_bundle_in_manifest


def load_stack_config(stack_path: Path | str) -> StackConfig:
    """Load and normalize the consolidated stack config index."""
    stack_file = Path(stack_path).resolve()
    root = stack_file.parents[1]
    doc = _load_yaml(stack_file)
    body = doc.get("rl_stack_locked", doc)
    if not isinstance(body, dict):
        raise ValueError("Missing `rl_stack_locked` mapping in stack config")

    raw_components = body.get("components", {})
    raw_seed_sets = body.get("seed_sets", {})
    if not isinstance(raw_components, dict) or not isinstance(raw_seed_sets, dict):
        raise ValueError("`components` and `seed_sets` must be mappings")

    components = {k: (root / str(v)).resolve() for k, v in raw_components.items()}
    seed_sets = {k: (root / str(v)).resolve() for k, v in raw_seed_sets.items()}
    spec_mismatch_policy, require_export_spec_bundle, persist_spec_bundle_in_manifest = _load_component_contract(
        components
    )

    return StackConfig(
        root=root,
        components=components,
        seed_sets=seed_sets,
        spec_mismatch_policy=spec_mismatch_policy,
        require_export_spec_bundle=require_export_spec_bundle,
        persist_spec_bundle_in_manifest=persist_spec_bundle_in_manifest,
    )
