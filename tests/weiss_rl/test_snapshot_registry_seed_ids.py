from __future__ import annotations

from pathlib import Path

from weiss_rl.training.snapshots import seed_snapshot_policy_id

from .snapshot_registry_test_support import _load_train_script_module


def test_seed_snapshot_policy_id_preserves_hash_input_and_sanitizes_policy_id() -> None:
    source_run_dir = Path("relative") / "seed_run"

    policy_id = seed_snapshot_policy_id(
        source_run_dir=source_run_dir,
        source_policy_id=" folder/policy\\000010 ",
    )

    assert policy_id == "seed_c3bd127559_folder_policy_000010"


def test_train_seed_snapshot_policy_id_wrapper_matches_training_helper() -> None:
    train_script = _load_train_script_module()
    source_run_dir = Path("relative") / "seed_run"

    assert train_script._seed_snapshot_policy_id(
        source_run_dir=source_run_dir,
        source_policy_id="policy/000020",
    ) == seed_snapshot_policy_id(source_run_dir=source_run_dir, source_policy_id="policy/000020")
