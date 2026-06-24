from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from weiss_rl.diagnostics.b2_audit import b2_disagreement_audit as audit_module

from .b2_disagreement_audit_test_support import REPO_ROOT, _make_record, _module_env


def test_load_matchup_source_extracts_unique_paired_seeds_in_first_seen_order(tmp_path: Path) -> None:
    episodes_path = tmp_path / "episodes.jsonl"
    records = [
        _make_record(pair_index=2, swap_index=0, episode_seed=42),
        _make_record(pair_index=2, swap_index=1, episode_seed=42),
        _make_record(pair_index=0, swap_index=0, episode_seed=7),
        _make_record(pair_index=0, swap_index=1, episode_seed=7),
        _make_record(pair_index=1, swap_index=0, episode_seed=42),
        _make_record(pair_index=1, swap_index=1, episode_seed=42),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    source = audit_module._load_matchup_source(episodes_path)

    assert source.focal_policy_id == "learner"
    assert source.opponent_policy_id == "B2 HeuristicPublic"
    assert source.paired_seeds == (42, 7)


def test_b2_disagreement_audit_requires_fixed_pythonhashseed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.diagnostics.b2_audit.b2_disagreement_audit",
            "--stack-config",
            "missing.yaml",
            "--run-dir",
            "runs/missing",
            "--output-run-dir",
            str(tmp_path / "audit"),
            "--episodes-jsonl",
            "missing_episodes.jsonl",
            "--policy-id",
            "policy_000001",
        ],
        cwd=REPO_ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a fixed PYTHONHASHSEED" in result.stderr


def test_b2_disagreement_audit_parser_accepts_b1_baseline_run_dir() -> None:
    args = audit_module._build_parser().parse_args(
        [
            "--stack-config",
            "configs/thesis/main.yaml",
            "--run-dir",
            "runs/main",
            "--output-run-dir",
            "runs/audit",
            "--episodes-jsonl",
            "runs/main/eval/b1/episodes.jsonl",
            "--policy-id",
            "policy_000002",
            "--opponent-policy-id",
            "B1 NoLeague baseline",
            "--b1-baseline-run-dir",
            "runs/locked_b1",
            "--require-opponent-context-index",
        ]
    )

    assert args.b1_baseline_run_dir == Path("runs/locked_b1")
    assert args.require_opponent_context_index is True
