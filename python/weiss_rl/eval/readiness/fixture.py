"""Compatibility facade for paper-readiness fixture generation."""

from __future__ import annotations

from weiss_rl.eval.readiness import fixture_writer as _writer

write_paper_readiness_run_fixture = _writer.write_paper_readiness_run_fixture
_write_diagnostics = _writer._write_diagnostics
_write_final_eval_tree = _writer._write_final_eval_tree
_write_json = _writer._write_json
_write_metagame_tree = _writer._write_metagame_tree
_write_paper_figures = _writer._write_paper_figures
_write_text = _writer._write_text

__all__ = ["write_paper_readiness_run_fixture"]
