from __future__ import annotations

import torch
from weiss_rl.training.script_entrypoint_hooks import run_periodic_dev_eval_with_script_hooks


def test_run_periodic_dev_eval_hook_accepts_new_and_legacy_stall_monitor_flag_names() -> None:
    captured: list[dict[str, object]] = []

    class Api:
        _PeriodicDevEvalRunner = object()
        _ensure_current_checkpoint = object()
        _current_focal_policy_id = object()
        _spec_dimensions = object()
        _clone_cpu_eval_model = object()
        _periodic_dev_eval_opponents = object()
        _persist_periodic_dev_eval_summary = object()
        _update_stall_monitor = object()
        _write_json = object()

        @staticmethod
        def run_periodic_dev_eval(**kwargs):
            captured.append(kwargs)
            return {"ok": True}

    base_kwargs = {
        "stack": object(),
        "contract": object(),
        "artifacts": object(),
        "training_paths": object(),
        "learner": object(),
        "device": torch.device("cpu"),
        "run_id256": "0" * 64,
        "config_hash256": "1" * 64,
        "spec_hash256": "2" * 64,
    }

    run_periodic_dev_eval_with_script_hooks(Api, **base_kwargs, update_stall_monitor=False)
    run_periodic_dev_eval_with_script_hooks(Api, **base_kwargs, update_stall_monitor_enabled=False)

    assert captured[0]["update_stall_monitor_enabled"] is False
    assert captured[1]["update_stall_monitor_enabled"] is False
