from __future__ import annotations

import weiss_rl.training.replay_data.paired_swing_conflict_filter as paired_swing_conflict_filter
import weiss_rl.training.replay_data.paired_swing_replay as paired_swing_replay


def test_paired_swing_replay_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "paired_swing_conflict_filter")
    assert not hasattr(training, "paired_swing_replay")


def test_paired_swing_replay_does_not_reexport_conflict_filter_helpers() -> None:
    retired_helper_exports = {
        "_normalize_action_source",
        "filter_paired_swing_conflict_rows",
        "normalize_paired_swing_action_source",
        "paired_swing_distinct_train_row_count",
    }

    assert paired_swing_replay.__all__ == [
        "PairedSwingReplayState",
        "maybe_run_paired_swing_replay",
    ]
    assert not any(hasattr(paired_swing_replay, name) for name in retired_helper_exports)
    assert callable(paired_swing_conflict_filter.filter_paired_swing_conflict_rows)
    assert callable(paired_swing_conflict_filter.paired_swing_distinct_train_row_count)
    assert callable(paired_swing_conflict_filter.normalize_paired_swing_action_source)
    assert (
        paired_swing_conflict_filter.filter_paired_swing_conflict_rows.__module__
        == "weiss_rl.training.replay_data.paired_swing_conflict_filter"
    )
