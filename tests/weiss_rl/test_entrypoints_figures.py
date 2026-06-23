from __future__ import annotations

from .entrypoints_test_support import (
    REPO_ROOT,
    Path,
    _run_entrypoint,
    _run_public_demo_train,
    json,
    os,
    public_demo_spec_hash256,
    subprocess,
    sys,
)


def test_make_figures_entrypoint_public_demo_writes_clearly_labeled_bundle(tmp_path: Path) -> None:
    train_result, run_dir = _run_public_demo_train(tmp_path, run_label="toy_public_demo_figures")
    assert train_result.returncode == 0, train_result.stderr
    stack_config = tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"
    eval_result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        extra_args=[
            "--public-demo",
            "--run-dir",
            str(run_dir),
            "--public-demo-paired-seeds",
            "4",
            "--public-demo-bootstrap-samples",
            "8",
        ],
    )
    assert eval_result.returncode == 0, eval_result.stderr

    figures_dir = run_dir / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.workflows.figures.figures_entrypoint",
            "--public-demo",
            "--final-eval-dir",
            str(run_dir / "eval" / "final_eval"),
            "--out-dir",
            str(figures_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    placeholder_path = figures_dir / "toy_demo_placeholder.txt"
    manifest_path = figures_dir / "toy_demo_manifest.json"
    assert placeholder_path.is_file()
    assert manifest_path.is_file()
    placeholder_text = placeholder_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "toy_public_demo_placeholder_figure" in placeholder_text
    assert manifest["demo_only"] is True
    assert manifest["public_safe"] is True
    assert "Wrote public-demo placeholder figure bundle" in result.stdout


def test_figure_mode_helpers_preserve_messages_and_default_paths(tmp_path: Path) -> None:
    from weiss_rl.workflows.figures.figure_modes import (
        run_paper_figure_mode,
        run_placeholder_figure_mode,
        run_public_demo_figure_mode,
    )

    observed: dict[str, object] = {}
    final_eval_dir = tmp_path / "runs" / "demo" / "eval" / "final_eval"

    def fake_public_demo_figures(*, final_eval_dir: Path, out_dir: Path) -> dict[str, Path]:
        observed["public_demo"] = (final_eval_dir, out_dir)
        return {"manifest": out_dir / "toy_demo_manifest.json"}

    def fake_placeholder(out: Path) -> None:
        observed["placeholder"] = out

    def fake_paper(run_dir: Path, *, formats: tuple[str, ...], fig_id: str | None) -> tuple[Path, ...]:
        observed["paper"] = (run_dir, formats, fig_id)
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    public_message = run_public_demo_figure_mode(
        final_eval_dir=final_eval_dir,
        out_dir=None,
        render_public_demo_figures_fn=fake_public_demo_figures,
    )
    placeholder_message = run_placeholder_figure_mode(
        out=tmp_path / "placeholder.txt",
        render_placeholder_figure_fn=fake_placeholder,
    )
    paper_message = run_paper_figure_mode(
        run_dir=tmp_path / "runs" / "main",
        formats=(),
        fig_id="seat_bias",
        render_paper_figures_fn=fake_paper,
    )

    assert observed["public_demo"] == (final_eval_dir, tmp_path / "runs" / "demo" / "figures")
    assert public_message.endswith("runs/demo/figures/toy_demo_manifest.json") or public_message.endswith(
        "runs\\demo\\figures\\toy_demo_manifest.json"
    )
    assert observed["placeholder"] == tmp_path / "placeholder.txt"
    assert placeholder_message == f"Wrote placeholder figure: {tmp_path / 'placeholder.txt'}"
    assert observed["paper"] == (tmp_path / "runs" / "main", ("pdf", "png"), "seat_bias")
    assert (
        paper_message == f"Wrote 1 files for fig-id 'seat_bias' to {tmp_path / 'runs' / 'main' / 'figures' / 'paper'}"
    )


def test_make_figures_entrypoint_exposes_only_cli_main() -> None:
    from weiss_rl.workflows.figures import figures_entrypoint

    retired_helper_exports = {
        "PAPER_FIGURE_IDS",
        "render_paper_figures",
        "render_placeholder_figure",
        "render_public_demo_figures",
        "run_paper_figure_mode",
        "run_placeholder_figure_mode",
        "run_public_demo_figure_mode",
    }

    assert figures_entrypoint.__all__ == ["main"]
    assert not any(hasattr(figures_entrypoint, name) for name in retired_helper_exports)
