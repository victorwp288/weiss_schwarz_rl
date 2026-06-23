from __future__ import annotations

from weiss_rl import replay as replay_package
from weiss_rl.replay import inspector as inspector_module
from weiss_rl.replay.inspector_report import (
    format_replay_inspection_report as report_format_replay_inspection_report,
)
from weiss_rl.replay.inspector_report import write_replay_inspection_report


def test_replay_inspector_report_helpers_are_package_owned() -> None:
    assert replay_package.format_replay_inspection_report is report_format_replay_inspection_report
    assert replay_package.write_replay_inspection_report is write_replay_inspection_report
    assert "format_replay_inspection_report" not in inspector_module.__all__
    assert "write_replay_inspection_report" not in inspector_module.__all__
    assert report_format_replay_inspection_report.__module__ == "weiss_rl.replay.inspector_report"
