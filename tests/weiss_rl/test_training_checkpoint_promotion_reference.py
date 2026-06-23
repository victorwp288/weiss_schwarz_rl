from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.minimal.promotion import TrainingCheckpointPromotionHooks, _league_reference_update_from_metrics

from .training_checkpoint_promotion_test_support import run_checkpoint_promotion


def test_league_reference_update_from_metrics_uses_effective_update_when_present() -> None:
    assert _league_reference_update_from_metrics({}) is None
    assert _league_reference_update_from_metrics({"league_effective_update": 42.0}) == 42


def test_checkpoint_promotion_skips_non_interval_update(tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_hook(**_kwargs: object) -> object:
        calls.append("unexpected")
        raise AssertionError("checkpoint hooks must not run outside the interval")

    tracker_payload = run_checkpoint_promotion(
        tmp_path=tmp_path,
        learner=SimpleNamespace(update_count=5, model=object(), get_policy_version=lambda: 9),
        runtime=SimpleNamespace(refresh_opponent_pool=fail_hook),
        checkpoint_interval_updates=3,
        hooks=TrainingCheckpointPromotionHooks(
            write_checkpoint=fail_hook,
            publish_checkpoint_aliases=fail_hook,
            maybe_log_structured_mainmove_guard=fail_hook,
            persist_snapshot_registry_entry=fail_hook,
            run_snapshot_promotion_gate=fail_hook,
        ),
    )

    assert tracker_payload is None
    assert calls == []
