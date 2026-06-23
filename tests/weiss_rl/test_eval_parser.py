from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import weiss_rl.workflows.eval_support.eval_parser as eval_parser
import weiss_rl.workflows.eval_support.eval_parser_arguments as eval_parser_arguments
import weiss_rl.workflows.eval_support.eval_parser_validation as eval_parser_validation
from weiss_rl.experiments.toy_public_demo import (
    PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
    PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
)


def test_eval_parser_exposes_only_parser_builder() -> None:
    retired_helper_exports = {
        "_require_positive_int",
        "_resolve_run_label",
        "add_canonical_eval_arguments",
        "add_eval_common_arguments",
        "add_public_demo_arguments",
        "add_summary_only_arguments",
    }

    assert eval_parser.__all__ == ["build_eval_parser"]
    assert not any(hasattr(eval_parser, name) for name in retired_helper_exports)
    assert callable(eval_parser_arguments.add_eval_common_arguments)
    assert callable(eval_parser_validation._resolve_run_label)


def test_eval_parser_preserves_canonical_public_demo_and_summary_arguments() -> None:
    parser = eval_parser.build_eval_parser()

    args = parser.parse_args(
        [
            "--stack-config",
            "configs/eval.yaml",
            "--spec-hash",
            "spec",
            "--config-hash",
            "config",
            "--run-label",
            "eval_label",
            "--run-id",
            "eval_label",
            "--public-demo",
            "--run-dir",
            "runs/main",
            "--final-eval-dir",
            "runs/main/eval/final_eval",
            "--policy-id",
            "B0 RandomLegal",
            "--policy-id",
            "policy_000100",
            "--snapshot-registry-json",
            "runs/source/training/snapshots/registry.json",
            "--dev-eval-summaries-json",
            "runs/source/training/logs/periodic_dev_eval_summaries.json",
            "--b1-baseline-run-dir",
            "runs/b1",
            "--paired-seed-limit",
            "9",
            "--stage1-paired-seeds",
            "3",
            "--max-paired-seeds",
            "11",
            "--skip-metagame",
            "--study-config",
            "configs/study/metagame.yaml",
            "--skip-figures",
            "--skip-readiness",
            "--git-commit-override",
            "a" * 40,
            "--public-demo-paired-seeds",
            "5",
            "--public-demo-bootstrap-samples",
            "17",
            "--episodes-jsonl",
            "runs/main/eval/final_eval/episodes.jsonl",
            "--summary-json",
            "summary.json",
            "--summary-csv",
            "summary.csv",
            "--diagnostics-json",
            "diagnostics.json",
            "--bootstrap-samples",
            "23",
            "--bootstrap-seed",
            "29",
        ]
    )

    assert args.stack_config == Path("configs/eval.yaml")
    assert args.spec_hash == "spec"
    assert args.config_hash == "config"
    assert args.run_label == "eval_label"
    assert args.run_id_alias == "eval_label"
    assert args.public_demo is True
    assert args.run_dir == Path("runs/main")
    assert args.final_eval_dir == Path("runs/main/eval/final_eval")
    assert args.policy_id == ["B0 RandomLegal", "policy_000100"]
    assert args.snapshot_registry_json == Path("runs/source/training/snapshots/registry.json")
    assert args.dev_eval_summaries_json == Path("runs/source/training/logs/periodic_dev_eval_summaries.json")
    assert args.b1_baseline_run_dir == Path("runs/b1")
    assert args.paired_seed_limit == 9
    assert args.stage1_paired_seeds == 3
    assert args.max_paired_seeds == 11
    assert args.skip_metagame is True
    assert args.study_config == Path("configs/study/metagame.yaml")
    assert args.skip_figures is True
    assert args.skip_readiness is True
    assert args.git_commit_override == "a" * 40
    assert args.public_demo_paired_seeds == 5
    assert args.public_demo_bootstrap_samples == 17
    assert args.episodes_jsonl == Path("runs/main/eval/final_eval/episodes.jsonl")
    assert args.summary_json == Path("summary.json")
    assert args.summary_csv == Path("summary.csv")
    assert args.diagnostics_json == Path("diagnostics.json")
    assert args.bootstrap_samples == 23
    assert args.bootstrap_seed == 29


def test_eval_parser_preserves_mode_defaults() -> None:
    parser = eval_parser.build_eval_parser()
    args = parser.parse_args(["--stack-config", "configs/eval.yaml"])

    assert args.public_demo is False
    assert args.run_label == ""
    assert args.run_id_alias == ""
    assert args.run_dir is None
    assert args.final_eval_dir is None
    assert args.policy_id is None
    assert args.snapshot_registry_json is None
    assert args.dev_eval_summaries_json is None
    assert args.b1_baseline_run_dir is None
    assert args.paired_seed_limit is None
    assert args.stage1_paired_seeds is None
    assert args.max_paired_seeds is None
    assert args.skip_metagame is False
    assert args.skip_figures is False
    assert args.skip_readiness is False
    assert args.git_commit_override == ""
    assert args.public_demo_paired_seeds == PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS
    assert args.public_demo_bootstrap_samples == PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES
    assert args.episodes_jsonl is None
    assert args.summary_json is None
    assert args.summary_csv is None
    assert args.diagnostics_json is None
    assert args.bootstrap_samples == 1000
    assert args.bootstrap_seed == 0


def test_eval_parser_validation_helpers_preserve_errors_and_alias_warning(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()

    assert eval_parser_validation._resolve_run_label(parser, " eval ", "eval") == "eval"
    assert "Warning: --run-id is deprecated; use --run-label instead." in capsys.readouterr().err
    assert eval_parser_validation._require_positive_int(parser, "--count", 4) == 4
    assert eval_parser_validation._require_positive_int(parser, "--count", None) is None

    with pytest.raises(SystemExit):
        eval_parser_validation._resolve_run_label(parser, "left", "right")
    with pytest.raises(SystemExit):
        eval_parser_validation._require_positive_int(parser, "--count", 0)
