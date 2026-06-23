from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import weiss_rl.eval.readiness.fixture_cli as paper_readiness_fixture_cli
import weiss_rl.eval.readiness.fixture_entrypoint as paper_readiness_fixture_entrypoint
import weiss_rl.eval.readiness.fixture_writer as paper_readiness_fixture_writer
from weiss_rl.eval import build_paper_readiness_summary

write_paper_readiness_run_fixture = paper_readiness_fixture_writer.write_paper_readiness_run_fixture


def test_paper_readiness_fixture_facade_is_removed() -> None:
    assert find_spec("weiss_rl.eval.readiness.fixture") is None


def test_paper_readiness_fixture_entrypoint_exposes_only_cli_boundary() -> None:
    assert hasattr(paper_readiness_fixture_entrypoint, "main")
    assert not hasattr(paper_readiness_fixture_entrypoint, "build_paper_readiness_fixture_parser")
    assert not hasattr(paper_readiness_fixture_entrypoint, "run_paper_readiness_fixture_command")


def test_paper_readiness_fixture_cli_parser_and_runtime_can_be_injected(tmp_path: Path) -> None:
    parser = paper_readiness_fixture_cli.build_paper_readiness_fixture_parser()
    requested_run_dir = tmp_path / "requested_run"
    written_run_dir = tmp_path / "written_run"
    args = parser.parse_args(["--run-dir", str(requested_run_dir)])
    observed: dict[str, Path] = {}

    def fake_write(run_dir: Path) -> Path:
        observed["run_dir"] = run_dir
        return written_run_dir

    result = paper_readiness_fixture_cli.run_paper_readiness_fixture_command(
        args,
        write_paper_readiness_run_fixture_fn=fake_write,
    )

    assert observed["run_dir"] == requested_run_dir
    assert result.run_dir == written_run_dir


def test_write_paper_readiness_run_fixture_satisfies_readiness_contract(tmp_path: Path) -> None:
    run_dir = write_paper_readiness_run_fixture(tmp_path / "run_ready")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
