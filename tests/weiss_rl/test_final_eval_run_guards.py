from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.eval import run_final_eval

from .final_eval_test_support import (
    _CONFIG_HASH256,
    _RUN_ID256,
    _SPEC_HASH256,
    _FakeMatrixRunner,
    _final_eval_stop_rules,
    _selection_config,
    _write_policy_set_inputs,
)


def test_run_final_eval_rejects_duplicate_explicit_policy_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_ids must be unique"):
        run_final_eval(
            output_dir=tmp_path / "final_eval",
            runner=_FakeMatrixRunner({}),
            policy_ids=["policy_dup", "policy_dup"],
            paired_seeds=[11],
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            stop_rules=_final_eval_stop_rules(),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
        )


def test_run_final_eval_rejects_underfilled_deterministic_selection(tmp_path: Path) -> None:
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    config = _selection_config(
        include_random_legal_baseline_b0=False,
        include_no_league_baseline_b1=False,
        include_heuristic_public_b2_if_exists=False,
        include_final_champion_snapshot=False,
        include_spaced_snapshots_near_percent_updates=(),
    )

    with pytest.raises(ValueError, match="underfilled"):
        run_final_eval(
            output_dir=tmp_path / "final_eval",
            runner=_FakeMatrixRunner({}),
            paired_seeds=[11],
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            stop_rules=_final_eval_stop_rules(),
            run_id256=_RUN_ID256,
            config_hash256=_CONFIG_HASH256,
            spec_hash256=_SPEC_HASH256,
            snapshot_registry_path=snapshot_registry_path,
            dev_eval_summaries_path=dev_eval_summaries_path,
            selection_config=config,
            final_policy_set_size=3,
        )
