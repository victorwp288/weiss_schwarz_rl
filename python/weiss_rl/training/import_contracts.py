from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.experiments.baselines import (
    canonical_config_sections as _canonical_config_sections,
)
from weiss_rl.experiments.baselines import (
    config_marks_noleague_baseline as _config_marks_noleague_baseline,
)
from weiss_rl.experiments.baselines import (
    is_noleague_baseline_role as _is_noleague_baseline_role,
)
from weiss_rl.experiments.baselines import (
    legacy_noleague_baseline_mode as _legacy_noleague_baseline_mode,
)
from weiss_rl.experiments.baselines import (
    role_from_config_canonical as _role_from_config_canonical,
)
from weiss_rl.experiments.baselines import (
    selected_candidate_is_locked_b1 as _selected_candidate_is_locked_b1,
)
from weiss_rl.models.state_dict_compat import state_dict_key_mismatch_for_context_compat
from weiss_rl.training.run_metadata import load_json_object

MODEL_IMPORT_COMPATIBILITY_IGNORED_KEYS = frozenset(
    {
        "opponent_context_policy_ids",
        "opponent_context_hidden_scale",
        "opponent_context_trainable_hidden_scale",
        "opponent_context_trainable_recurrent_scale",
        "opponent_context_trainable_action_bias_scale",
        "opponent_context_trainable_candidate_residual_scale",
        "opponent_context_candidate_residual_width",
        "opponent_context_candidate_residual_mode",
        "opponent_context_candidate_residual_action_ids",
        "opponent_context_adapter_lr_multiplier",
        "opponent_context_adapter_train_only",
        "opponent_context_eval_policy_ids",
    }
)


def is_noleague_baseline_role(role: str) -> bool:
    return _is_noleague_baseline_role(role)


def canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    return _canonical_config_sections(config_canonical)


def role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    return _role_from_config_canonical(config_canonical)


def legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    return _legacy_noleague_baseline_mode(config_canonical)


def config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    return _config_marks_noleague_baseline(config_canonical)


def assert_noleague_baseline_config(config_canonical: Mapping[str, Any]) -> None:
    role = role_from_config_canonical(config_canonical)
    if role:
        if not is_noleague_baseline_role(role):
            raise RuntimeError(
                f"Imported B1 baseline must come from a dedicated baseline_noleague run, got experiment.role={role!r}"
            )
        return
    legacy_mode = legacy_noleague_baseline_mode(config_canonical)
    if legacy_mode and legacy_mode != "b1_no_league":
        raise RuntimeError(
            "Imported B1 baseline must come from a dedicated baseline_noleague run, "
            f"got training_family_a.mode={legacy_mode!r}"
        )


def read_optional_hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def normalize_config_section_for_import_contract(section_name: str, section: Any) -> Any:
    if section_name != "model" or not isinstance(section, Mapping):
        return section
    normalized = dict(section)
    for key in MODEL_IMPORT_COMPATIBILITY_IGNORED_KEYS:
        normalized.pop(key, None)
    return normalized


def config_sections_match_for_import_contract(
    *,
    section_name: str,
    source_section: Any,
    expected_section: Any,
) -> bool:
    return normalize_config_section_for_import_contract(
        section_name,
        source_section,
    ) == normalize_config_section_for_import_contract(
        section_name,
        expected_section,
    )


def validate_imported_snapshot_contract(
    *,
    source_run_dir: Path,
    source_policy_id: str | None = None,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    source_layout = ArtifactLayout.from_run_dir(source_run_dir)
    manifest_path = source_layout.manifest_path
    source_manifest = load_json_object(manifest_path, label="imported B1 manifest") if manifest_path.is_file() else None
    source_config_canonical = source_manifest.get("config_canonical") if isinstance(source_manifest, dict) else None
    if isinstance(source_config_canonical, dict):
        source_config_sections = canonical_config_sections(source_config_canonical)
        if not _selected_candidate_is_locked_b1_source(
            source_run_dir=source_run_dir,
            source_policy_id=source_policy_id,
        ):
            assert_noleague_baseline_config(source_config_canonical)
        if isinstance(expected_config_canonical, dict):
            expected_config_sections = canonical_config_sections(expected_config_canonical)
            for section_name in ("model", "environment"):
                source_section = source_config_sections.get(section_name)
                expected_section = expected_config_sections.get(section_name)
                if source_section is None or expected_section is None:
                    continue
                if not config_sections_match_for_import_contract(
                    section_name=section_name,
                    source_section=source_section,
                    expected_section=expected_section,
                ):
                    raise RuntimeError(
                        f"Imported B1 baseline config does not match the current run for section={section_name!r}"
                    )

    if expected_spec_hash256 is not None:
        source_spec_hash = read_optional_hash_file(source_layout.spec_hash_path)
        if source_spec_hash is not None and source_spec_hash != expected_spec_hash256:
            raise RuntimeError(
                "Imported B1 baseline spec hash does not match the current run: "
                f"source={source_spec_hash} expected={expected_spec_hash256}"
            )

    source_model_state_dict = payload.get("model_state_dict")
    if not isinstance(source_model_state_dict, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload is missing model_state_dict: {source_run_dir}")
    missing, extra, _allowed_missing = state_dict_key_mismatch_for_context_compat(
        source_state_dict=source_model_state_dict,
        expected_state_dict=expected_model_state_dict,
    )
    if missing or extra:
        raise RuntimeError(
            "Imported B1 baseline model contract does not match the current run: "
            f"missing_keys={missing} extra_keys={extra}"
        )
    for key in sorted(set(source_model_state_dict) & set(expected_model_state_dict)):
        source_value = source_model_state_dict[key]
        expected_value = expected_model_state_dict[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(expected_value, torch.Tensor):
            continue
        if tuple(source_value.shape) != tuple(expected_value.shape) or source_value.dtype != expected_value.dtype:
            raise RuntimeError(
                "Imported B1 baseline tensor contract does not match the current run: "
                f"key={key} source_shape={tuple(source_value.shape)} "
                f"expected_shape={tuple(expected_value.shape)} "
                f"source_dtype={source_value.dtype} expected_dtype={expected_value.dtype}"
            )


def _selected_candidate_is_locked_b1_source(*, source_run_dir: Path, source_policy_id: str | None) -> bool:
    if str(source_policy_id or "").strip() != "selected_candidate":
        return False
    return _selected_candidate_is_locked_b1(source_run_dir)
