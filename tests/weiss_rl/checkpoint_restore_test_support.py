from __future__ import annotations

import torch
from weiss_rl.training.checkpointing.storage.restore import CheckpointPayloadContract


class RestoreModelDouble:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.loaded_state = state_dict


class RestoreOptimizerDouble:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


class RestoreGradScalerDouble:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


class OrderedRestoreModelDouble:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._events.append("model")
        self.loaded_state = state_dict


class OrderedRestoreOptimizerDouble:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self._events.append("optimizer")
        self.loaded_state = state_dict


class OrderedRestoreGradScalerDouble:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self._events.append("grad_scaler")
        self.loaded_state = state_dict


class ResumeLearnerDouble:
    def __init__(self) -> None:
        self.model = RestoreModelDouble()
        self.optimizer = RestoreOptimizerDouble()
        self._grad_scaler = RestoreGradScalerDouble()
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> RestoreOptimizerDouble:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.loaded_anchor_state = state_dict


class OrderedResumeLearnerDouble:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.model = OrderedRestoreModelDouble(self.events)
        self.optimizer = OrderedRestoreOptimizerDouble(self.events)
        self._grad_scaler = OrderedRestoreGradScalerDouble(self.events)
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> OrderedRestoreOptimizerDouble:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.events.append("anchor")
        self.loaded_anchor_state = state_dict


class InitLearnerWithoutResetDouble:
    def __init__(self) -> None:
        self.model = RestoreModelDouble()
        self.optimizer = RestoreOptimizerDouble()
        self._grad_scaler = RestoreGradScalerDouble()
        self.update_count = 99
        self.policy_version = 88
        self.total_samples_processed = 77
        self.loaded_anchor_state: object = "not-called"

    def _optimizer_for_step(self) -> RestoreOptimizerDouble:
        return self.optimizer

    def load_policy_anchor_state_dict(self, state_dict: object) -> None:
        self.loaded_anchor_state = state_dict


def minimal_checkpoint_contract(payload: dict[str, object] | None = None) -> CheckpointPayloadContract:
    full_payload: dict[str, object] = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": 3,
        "policy_version": 7,
        "total_samples_processed": 42,
        "init_schedule_offset_updates": 5,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "policy_anchor_model_state_dict": {"anchor_weight": torch.tensor([2.0])},
        "optimizer_state_dict": {"lr": 0.01},
        "grad_scaler_state_dict": {"scale": 2.0},
    }
    if payload is not None:
        full_payload.update(payload)
    return CheckpointPayloadContract(
        payload=full_payload,
        model_state_dict=full_payload["model_state_dict"],  # type: ignore[arg-type]
        config_hash_mismatch=False,
        expected_config_hash="abc",
        payload_config_hash="abc",
    )
