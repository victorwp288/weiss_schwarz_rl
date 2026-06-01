"""Runtime orchestration for paired-outcome preference span audits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.paired_outcome_preference_span_audit import (
    PairedOutcomePreferenceSpanAuditConfig,
    build_paired_outcome_preference_span_audit,
    write_paired_outcome_preference_span_audit,
)


@dataclass(frozen=True, slots=True)
class PairedOutcomePreferenceSpanAuditRunResult:
    output_json: Path
    report: dict[str, Any]


def run_paired_outcome_preference_span_audit(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSpanAuditRunResult:
    report = build_paired_outcome_preference_span_audit(paired_outcome_preference_span_audit_config_from_args(args))
    write_paired_outcome_preference_span_audit(args.output_json, report)
    return PairedOutcomePreferenceSpanAuditRunResult(output_json=args.output_json, report=report)


def paired_outcome_preference_span_audit_config_from_args(
    args: argparse.Namespace,
) -> PairedOutcomePreferenceSpanAuditConfig:
    return PairedOutcomePreferenceSpanAuditConfig(
        dataset_path=args.dataset.resolve(),
        spec_bundle_json=None if args.spec_bundle_json is None else args.spec_bundle_json.resolve(),
        max_gap=int(args.max_gap),
        max_compact_span_width=int(args.max_compact_span_width),
        min_repeated_pair_count=int(args.min_repeated_pair_count),
        max_examples=int(args.max_examples),
    )


__all__ = [
    "PairedOutcomePreferenceSpanAuditRunResult",
    "paired_outcome_preference_span_audit_config_from_args",
    "run_paired_outcome_preference_span_audit",
]
