from __future__ import annotations

import json

import numpy as np
from weiss_rl.actors.actor_faults import write_actor_numeric_fault_bundle


def test_write_actor_numeric_fault_bundle_records_optional_nonfinite_fields(tmp_path) -> None:
    fault_path, payload = write_actor_numeric_fault_bundle(
        fault_dir=tmp_path,
        reason="bad actor values",
        actor_id=3,
        layout_name="mask",
        update_count=7,
        observed_checkpoint_update=5,
        step=2,
        obs=np.array([[1, 2]], dtype=np.int16),
        to_play=np.array([0], dtype=np.int8),
        decision_id=np.array([11], dtype=np.int32),
        episode_seed=np.array([101], dtype=np.uint64),
        episode_key=np.array([201], dtype=np.uint64),
        logits=np.array([[0.0, np.nan]], dtype=np.float32),
        actions=np.array([1], dtype=np.int64),
        logp=np.array([np.inf], dtype=np.float32),
        entropy=np.array([np.nan], dtype=np.float32),
        legal_mask=np.array([[True, True]], dtype=np.bool_),
    )

    assert fault_path.name.startswith("actor_numeric_fault_")
    assert payload["component"] == "actor_worker"
    assert payload["reason"] == "bad actor values"

    serialized = json.loads(fault_path.read_text(encoding="utf-8"))
    assert serialized["logits_nonfinite_indices"]["data"] == [[0, 1]]
    assert serialized["logp_nonfinite_indices"]["data"] == [[0]]
    assert serialized["entropy_nonfinite_indices"]["data"] == [[0]]
    assert serialized["legal_mask"]["data"] == [[True, True]]
