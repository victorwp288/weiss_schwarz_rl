from __future__ import annotations

import importlib.util


def test_weiss_rl_tests_is_not_an_importable_package_surface() -> None:
    assert importlib.util.find_spec("weiss_rl.tests") is None


def test_runtime_shared_memory_compat_facade_is_not_importable() -> None:
    assert importlib.util.find_spec("weiss_rl.runtime.components.shared") is None


def test_replay_package_exports_only_inspection_surface() -> None:
    import weiss_rl.replay as replay_package

    assert replay_package.__all__ == [
        "format_replay_inspection_report",
        "inspect_replay_bundle",
        "write_replay_inspection_report",
    ]
    assert not hasattr(replay_package, "ReplayStep")
    assert not hasattr(replay_package, "ReplayTrajectoryDataset")
    assert not hasattr(replay_package, "verify_replay_bundle")
