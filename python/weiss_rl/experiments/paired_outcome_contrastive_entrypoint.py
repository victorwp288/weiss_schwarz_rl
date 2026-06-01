from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.experiments import paired_outcome_contrastive as _contrastive
from weiss_rl.experiments.paired_outcome_contrastive_cli import build_paired_outcome_contrastive_parser
from weiss_rl.experiments.paired_outcome_contrastive_reporting import paired_outcome_contrastive_output_line
from weiss_rl.experiments.paired_outcome_contrastive_runtime import run_paired_outcome_contrastive_dataset

PairedOutcomeContrastiveBuildConfig = _contrastive.PairedOutcomeContrastiveBuildConfig
PairedOutcomeContrastiveSource = _contrastive.PairedOutcomeContrastiveSource
PairedOutcomeInspectionConfig = _contrastive.PairedOutcomeInspectionConfig
PairedOutcomeInspectionSource = _contrastive.PairedOutcomeInspectionSource
build_paired_outcome_contrastive_dataset = _contrastive.build_paired_outcome_contrastive_dataset
inspect_paired_outcome_sources = _contrastive.inspect_paired_outcome_sources
sources_from_paired_flip_summary = _contrastive.sources_from_paired_flip_summary
write_paired_outcome_contrastive_summary = _contrastive.write_paired_outcome_contrastive_summary
_build_parser = build_paired_outcome_contrastive_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_paired_outcome_contrastive_parser().parse_args(argv)
    result = run_paired_outcome_contrastive_dataset(args)
    print(
        paired_outcome_contrastive_output_line(
            output_dataset=result.output_dataset,
            summary_path=result.summary_path,
            dataset=result.dataset,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
