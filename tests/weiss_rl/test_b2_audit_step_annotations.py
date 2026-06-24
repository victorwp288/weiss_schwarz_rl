from __future__ import annotations

from pathlib import Path

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.diagnostics.b2_audit import b2_disagreement_audit as audit_module


def test_annotate_step_diff_uses_action_catalog_decoder() -> None:
    decoder = ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_space_size": 7,
                "pass_action_id": 0,
                "constants": [["MAX_HAND", 50], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "attack_type_encoding": [["front", 0], ["side", 1]],
                "families": [
                    {"name": "pass", "base": 0, "count": 1},
                    {"name": "attack", "base": 1, "count": 6},
                ],
            }
        }
    )

    annotated = audit_module._annotate_step_diff(
        {
            "recorded_action": 0,
            "policy_a_top_action": {"action": 0},
            "policy_b_top_action": {"action": 1},
        },
        decoder=decoder,
    )

    assert annotated["recorded_action_family"] == "pass"
    assert annotated["policy_a_top_action_family"] == "pass"
    assert annotated["policy_b_top_action_family"] == "attack"


def test_materialize_audit_bundle_copy_uses_pair_and_swap_suffix(tmp_path: Path) -> None:
    source_bundle = tmp_path / "replays" / "bundles" / "replay_deadbeef.zip"
    source_bundle.parent.mkdir(parents=True, exist_ok=True)
    source_bundle.write_text("bundle-bytes", encoding="utf-8")
    bundle_copies_dir = tmp_path / "audit" / "replay_bundles"
    bundle_copies_dir.mkdir(parents=True, exist_ok=True)

    copied = audit_module._materialize_audit_bundle_copy(
        source_bundle_path=source_bundle,
        bundle_copies_dir=bundle_copies_dir,
        pair_index=7,
        swap_index=1,
    )

    assert copied == bundle_copies_dir / "replay_deadbeef_pair007_swap1.zip"
    assert copied.read_text(encoding="utf-8") == "bundle-bytes"
