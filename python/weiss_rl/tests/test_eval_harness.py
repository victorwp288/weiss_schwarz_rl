from __future__ import annotations

import json

import numpy as np
import pytest

from weiss_rl.eval.harness import abort_on_engine_fault_eval


def test_abort_on_engine_fault_eval_writes_artifact_and_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="engine_status!=0 during evaluation"):
        abort_on_engine_fault_eval(
            run_dir=tmp_path,
            engine_status=np.array([0, 9, 4], dtype=np.int32),
            decision_id=np.array([100, 101, 102], dtype=np.int64),
            episode_key=b"episode-7",
        )

    payload = json.loads((tmp_path / "eval_engine_fault.json").read_text(encoding="utf-8"))
    assert payload == {
        "decision_id": [100, 101, 102],
        "engine_status": [0, 9, 4],
        "episode_key": "b'episode-7'",
        "fault_env_indices": [1, 2],
        "note": "engine_status!=0 during evaluation",
    }


def test_abort_on_engine_fault_eval_is_a_noop_without_faults(tmp_path) -> None:
    abort_on_engine_fault_eval(
        run_dir=tmp_path,
        engine_status=np.array([0, 0], dtype=np.int32),
        decision_id=np.array([1, 2], dtype=np.int64),
    )

    assert not (tmp_path / "eval_engine_fault.json").exists()
