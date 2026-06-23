from __future__ import annotations

from types import SimpleNamespace

import weiss_rl.training.loop.update_completion as training_update_completion


def test_merge_post_update_auxiliary_metrics_into_training_log_uses_latest_learner_record() -> None:
    calls: list[dict[str, object]] = []
    logger = SimpleNamespace(
        merge_latest_custom_metrics=lambda **kwargs: calls.append(dict(kwargs)),
    )
    learner = SimpleNamespace(
        logger=logger,
        update_count=7,
        get_policy_version=lambda: 3,
    )
    metrics = {"paired_swing_replay_loss": 0.25}

    training_update_completion.merge_post_update_auxiliary_metrics_into_training_log(learner=learner, metrics=metrics)

    assert calls == [
        {
            "update_count": 7,
            "policy_version": 3,
            "metrics": metrics,
            "prefixes": training_update_completion.POST_UPDATE_TRAINING_LOG_METRIC_PREFIXES,
        }
    ]
    assert "pfsp_" in calls[0]["prefixes"]
    assert "collector_pfsp_" in calls[0]["prefixes"]
