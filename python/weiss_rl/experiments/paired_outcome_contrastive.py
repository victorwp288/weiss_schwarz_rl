"""Compatibility facade for paired-outcome contrastive replay datasets."""

from __future__ import annotations

from typing import Any

from weiss_rl.experiments import paired_outcome_contrastive_inspection as _inspection
from weiss_rl.experiments.paired_outcome_contrastive_build import (
    SCRIPT_KIND,
    PairedOutcomeContrastiveBuildConfig,
    apply_policy_b_top_action_overrides,
    build_paired_outcome_contrastive_dataset,
    build_paired_outcome_contrastive_source_dataset,
    write_paired_outcome_contrastive_summary,
)
from weiss_rl.experiments.paired_outcome_contrastive_inspection import PairedOutcomeInspectionConfig
from weiss_rl.experiments.paired_outcome_contrastive_sources import (
    PairedOutcomeContrastiveSource,
    PairedOutcomeInspectionSource,
    sources_from_paired_flip_summary,
)

inspect_replay_bundle = _inspection.inspect_replay_bundle
write_replay_inspection_report = _inspection.write_replay_inspection_report


def inspect_paired_outcome_sources(
    config: PairedOutcomeInspectionConfig,
) -> tuple[tuple[PairedOutcomeContrastiveSource, ...], dict[str, Any]]:
    _inspection.inspect_replay_bundle = inspect_replay_bundle
    _inspection.write_replay_inspection_report = write_replay_inspection_report
    return _inspection.inspect_paired_outcome_sources(config)


__all__ = [
    "PairedOutcomeContrastiveBuildConfig",
    "PairedOutcomeContrastiveSource",
    "PairedOutcomeInspectionConfig",
    "PairedOutcomeInspectionSource",
    "SCRIPT_KIND",
    "apply_policy_b_top_action_overrides",
    "build_paired_outcome_contrastive_dataset",
    "build_paired_outcome_contrastive_source_dataset",
    "inspect_paired_outcome_sources",
    "sources_from_paired_flip_summary",
    "write_paired_outcome_contrastive_summary",
]
