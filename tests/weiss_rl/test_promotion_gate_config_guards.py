from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.config import load_stack_config
from weiss_rl.league import run_promotion_gate

from ._config_paths import canonical_stack_config_path
from .promotion_gate_test_support import (
    CONFIG_HASH256,
    RUN_ID256,
    SPEC_HASH256,
    OutcomeRunner,
    stack_with_promotion_gate_override,
)


def test_run_promotion_gate_rejects_unexpected_uncertainty_method(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())
    stack = stack_with_promotion_gate_override(stack, uncertainty_method="jackknife_v1")

    with pytest.raises(
        ValueError,
        match="promotion gate requires uncertainty_method='bayesian_bootstrap_seedlevel_v1', got 'jackknife_v1'",
    ):
        run_promotion_gate(
            stack=stack,
            run_dir=tmp_path,
            focal_policy_id="candidate_snapshot",
            anchor_policy_ids={
                "B0 RandomLegal": "random_legal",
                "B1 NoLeague baseline": "b1_snapshot",
            },
            runner=OutcomeRunner(lambda _: "W"),
            run_id256=RUN_ID256,
            config_hash256=CONFIG_HASH256,
            spec_hash256=SPEC_HASH256,
            sample_count=32,
            bootstrap_seed=5,
        )


def test_run_promotion_gate_rejects_missing_required_anchor(tmp_path: Path) -> None:
    stack = load_stack_config(canonical_stack_config_path())

    with pytest.raises(ValueError, match="missing required promotion anchors: B1 NoLeague baseline"):
        run_promotion_gate(
            stack=stack,
            run_dir=tmp_path,
            focal_policy_id="candidate_snapshot",
            anchor_policy_ids={"B0 RandomLegal": "random_legal"},
            runner=OutcomeRunner(lambda _: "W"),
            run_id256=RUN_ID256,
            config_hash256=CONFIG_HASH256,
            spec_hash256=SPEC_HASH256,
            sample_count=32,
            bootstrap_seed=5,
        )
