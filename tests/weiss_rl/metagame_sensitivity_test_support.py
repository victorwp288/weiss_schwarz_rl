from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import run_final_eval

from .final_eval_test_support import _CONFIG_HASH256, _RUN_ID256, _SPEC_HASH256, _FakeMatrixRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG_PATH = REPO_ROOT / "configs" / "study" / "metagame_sensitivity.yaml"


def write_final_eval_fixture(tmp_path: Path) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = ["policy_gamma", "policy_alpha", "policy_beta"]
    outcomes: dict[tuple[str, str, int, int], Literal["W", "L", "D", "T"]] = {
        ("policy_gamma", "policy_gamma", 0, 0): "D",
        ("policy_gamma", "policy_gamma", 0, 1): "D",
        ("policy_gamma", "policy_gamma", 1, 0): "D",
        ("policy_gamma", "policy_gamma", 1, 1): "D",
        ("policy_gamma", "policy_alpha", 0, 0): "W",
        ("policy_gamma", "policy_alpha", 0, 1): "T",
        ("policy_gamma", "policy_alpha", 1, 0): "W",
        ("policy_gamma", "policy_alpha", 1, 1): "L",
        ("policy_gamma", "policy_beta", 0, 0): "L",
        ("policy_gamma", "policy_beta", 0, 1): "L",
        ("policy_gamma", "policy_beta", 1, 0): "L",
        ("policy_gamma", "policy_beta", 1, 1): "W",
        ("policy_alpha", "policy_alpha", 0, 0): "D",
        ("policy_alpha", "policy_alpha", 0, 1): "D",
        ("policy_alpha", "policy_alpha", 1, 0): "D",
        ("policy_alpha", "policy_alpha", 1, 1): "D",
        ("policy_alpha", "policy_beta", 0, 0): "W",
        ("policy_alpha", "policy_beta", 0, 1): "W",
        ("policy_alpha", "policy_beta", 1, 0): "W",
        ("policy_alpha", "policy_beta", 1, 1): "L",
        ("policy_beta", "policy_beta", 0, 0): "D",
        ("policy_beta", "policy_beta", 0, 1): "D",
        ("policy_beta", "policy_beta", 1, 0): "D",
        ("policy_beta", "policy_beta", 1, 1): "D",
    }
    run_final_eval(
        output_dir=final_eval_dir,
        runner=_FakeMatrixRunner(outcomes),
        policy_ids=policies,
        paired_seeds=[11, 22],
        stage1_paired_seeds=2,
        max_paired_seeds=2,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=8,
    )
    return final_eval_dir


def write_final_eval_summary(final_eval_dir: Path, payload: dict[str, Any]) -> None:
    (final_eval_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
