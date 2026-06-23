from __future__ import annotations

from weiss_rl.training.minimal.finalization import _final_dev_eval_summary_for_update


def test_final_dev_eval_summary_for_update_uses_only_current_update_summary() -> None:
    summary = {"aggregate_score": 0.75}

    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=summary,
            last_dev_eval_update_count=12,
            learner_update_count=12,
        )
        is summary
    )
    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=summary,
            last_dev_eval_update_count=11,
            learner_update_count=12,
        )
        is None
    )
    assert (
        _final_dev_eval_summary_for_update(
            last_dev_eval_summary=None,
            last_dev_eval_update_count=12,
            learner_update_count=12,
        )
        is None
    )
