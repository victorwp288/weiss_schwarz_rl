#!/usr/bin/env python3
"""Report compact trajectory spans in paired-outcome preference replay datasets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_preference_span_audit as _span_audit
from weiss_rl.experiments.paired_outcome_preference_span_audit_cli import (
    build_paired_outcome_preference_span_audit_parser,
    parse_paired_outcome_preference_span_audit_args,
)
from weiss_rl.experiments.paired_outcome_preference_span_audit_reporting import (
    paired_outcome_preference_span_audit_output_line,
)
from weiss_rl.experiments.paired_outcome_preference_span_audit_runtime import (
    run_paired_outcome_preference_span_audit,
)

PairedOutcomePreferenceSpanAuditConfig = _span_audit.PairedOutcomePreferenceSpanAuditConfig
build_paired_outcome_preference_span_audit = _span_audit.build_paired_outcome_preference_span_audit
write_paired_outcome_preference_span_audit = _span_audit.write_paired_outcome_preference_span_audit
_build_parser = build_paired_outcome_preference_span_audit_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_paired_outcome_preference_span_audit_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_paired_outcome_preference_span_audit(parse_args(argv))
    print(paired_outcome_preference_span_audit_output_line(output_json=result.output_json, report=result.report))


if __name__ == "__main__":
    main()
