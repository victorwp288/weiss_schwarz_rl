from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_discovery import (
    load_b1_dev_eval_records as load_b1_dev_eval_records,
)
from weiss_rl.experiments.b1_candidate_payloads import json_object
from weiss_rl.experiments.b1_candidate_payloads import (
    load_reference_anchor_scores as load_reference_anchor_scores,
)
from weiss_rl.experiments.b1_candidate_publish import (
    B1_CANDIDATE_ALIAS_METADATA_FORMAT,
    SELECTED_CANDIDATE_ALIAS_METADATA_FORMAT,
    publish_snapshot_alias,
)
from weiss_rl.experiments.b1_candidate_report import (
    DEFAULT_CONFIRM_OPPONENTS as DEFAULT_CONFIRM_OPPONENTS,
)
from weiss_rl.experiments.b1_candidate_report import (
    DEFAULT_REQUIRED_ANCHORS as DEFAULT_REQUIRED_ANCHORS,
)
from weiss_rl.experiments.b1_candidate_report import (
    build_b1_candidate_selection as build_b1_candidate_selection,
)
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID, config_marks_noleague_baseline
from weiss_rl.league.registry import REGISTRY_FILENAME

SELECTED_CANDIDATE_POLICY_ID = "selected_candidate"


def publish_b1_baseline_alias(
    *,
    run_dir: Path,
    source_policy_id: str,
    selection_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    registry_path = run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    manifest = json_object(run_dir / "manifest.json")
    config_canonical = None if manifest is None else manifest.get("config_canonical")
    if not isinstance(config_canonical, Mapping) or not config_marks_noleague_baseline(config_canonical):
        raise RuntimeError(
            "Refusing to publish b1_noleague_baseline from a run that is not marked experiment.role='baseline_noleague'"
        )
    return publish_snapshot_alias(
        run_dir=run_dir,
        source_policy_id=source_policy_id,
        alias_policy_id=NOLEAGUE_BASELINE_POLICY_ID,
        metadata_format=B1_CANDIDATE_ALIAS_METADATA_FORMAT,
        selection_summary=selection_summary,
    )


def publish_selected_candidate_alias(
    *,
    run_dir: Path,
    source_policy_id: str,
    alias_policy_id: str = SELECTED_CANDIDATE_POLICY_ID,
    selection_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    normalized_alias = str(alias_policy_id).strip()
    if not normalized_alias:
        raise ValueError("alias_policy_id must be non-empty")
    if normalized_alias == NOLEAGUE_BASELINE_POLICY_ID:
        raise ValueError("use publish_b1_baseline_alias for the canonical B1 baseline alias")
    return publish_snapshot_alias(
        run_dir=run_dir,
        source_policy_id=source_policy_id,
        alias_policy_id=normalized_alias,
        metadata_format=SELECTED_CANDIDATE_ALIAS_METADATA_FORMAT,
        selection_summary=selection_summary,
        skip_copy_if_same_path=True,
    )
