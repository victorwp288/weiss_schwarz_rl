from __future__ import annotations

import torch
from weiss_rl.config import load_stack_config
from weiss_rl.training.manifest_layout import hardware_actor_layout_payload, manifest_actor_device_layout

from ._config_paths import repo_root


def test_manifest_actor_device_layout_matches_runtime_device_names() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_thesis_locked.yaml")

    layout = manifest_actor_device_layout(
        stack=stack,
        num_envs=2,
        unroll_length=4,
        profile="typed",
        seed=123,
        pass_action_id=0,
        runtime_mode="train_ordered",
        learner_device=torch.device("cpu"),
    )

    assert layout is not None
    assert all(isinstance(device_name, str) for device_name in layout)
    assert len(layout) >= 1


def test_hardware_actor_layout_payload_normalizes_device_names() -> None:
    assert hardware_actor_layout_payload(None) is None
    assert hardware_actor_layout_payload(()) is None
    assert hardware_actor_layout_payload((torch.device("cpu"), "cuda:0")) == ("cpu", "cuda:0")
