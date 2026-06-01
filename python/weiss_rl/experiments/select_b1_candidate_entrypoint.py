from __future__ import annotations

from weiss_rl.experiments import b1_candidate_selection as _selection
from weiss_rl.experiments import select_b1_candidate_runtime as _runtime
from weiss_rl.experiments.select_b1_candidate_cli import build_select_b1_candidate_parser
from weiss_rl.experiments.select_b1_candidate_reporting import (
    command_text,
    select_b1_candidate_output_lines,
)

DEFAULT_CONFIRM_OPPONENTS = _selection.DEFAULT_CONFIRM_OPPONENTS
DEFAULT_REQUIRED_ANCHORS = _selection.DEFAULT_REQUIRED_ANCHORS
SELECTED_CANDIDATE_POLICY_ID = _selection.SELECTED_CANDIDATE_POLICY_ID
build_b1_candidate_selection = _selection.build_b1_candidate_selection
load_reference_anchor_scores = _selection.load_reference_anchor_scores
publish_b1_baseline_alias = _selection.publish_b1_baseline_alias
publish_selected_candidate_alias = _selection.publish_selected_candidate_alias
_build_parser = build_select_b1_candidate_parser
_command_text = command_text
run_select_b1_candidate = _runtime.run_select_b1_candidate
_baseline_alias_selection_summary = _runtime.baseline_alias_selection_summary
_selected_alias_selection_summary = _runtime.selected_alias_selection_summary


def main() -> None:
    result = run_select_b1_candidate(build_select_b1_candidate_parser().parse_args())
    for line in select_b1_candidate_output_lines(result.summary, output_json=result.output_json):
        print(line)


if __name__ == "__main__":
    main()
