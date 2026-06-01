from __future__ import annotations

from pathlib import Path

from weiss_rl.eval import (
    build_paper_readiness_summary,
    paper_readiness_fixture,
    paper_readiness_fixture_cli,
    paper_readiness_fixture_entrypoint,
    paper_readiness_fixture_writer,
)

write_paper_readiness_run_fixture = paper_readiness_fixture.write_paper_readiness_run_fixture


def test_paper_readiness_fixture_facade_keeps_writer_aliases() -> None:
    assert paper_readiness_fixture.write_paper_readiness_run_fixture is (
        paper_readiness_fixture_writer.write_paper_readiness_run_fixture
    )
    assert paper_readiness_fixture._write_final_eval_tree is paper_readiness_fixture_writer._write_final_eval_tree
    assert paper_readiness_fixture._write_metagame_tree is paper_readiness_fixture_writer._write_metagame_tree


def test_paper_readiness_fixture_entrypoint_keeps_cli_aliases() -> None:
    assert paper_readiness_fixture_entrypoint.build_paper_readiness_fixture_parser is (
        paper_readiness_fixture_cli.build_paper_readiness_fixture_parser
    )
    assert paper_readiness_fixture_entrypoint.run_paper_readiness_fixture_command is (
        paper_readiness_fixture_cli.run_paper_readiness_fixture_command
    )


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
