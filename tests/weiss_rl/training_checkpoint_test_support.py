from __future__ import annotations

from pathlib import Path

import torch


class _Model:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Optimizer:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"lr": 0.01}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _GradScaler:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"scale": 2.0}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Learner:
    update_count = 3
    total_samples_processed = 42

    def __init__(self, *, model=...) -> None:
        self.model = _Model() if model is ... else model
        self.optimizer = _Optimizer()
        self._grad_scaler = _GradScaler()
        self.policy_version = 7
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.anchor_state: dict[str, torch.Tensor] | None = None
        self.loaded_anchor_state: dict[str, torch.Tensor] | None = None
        self.reset_anchor_calls = 0

    def get_policy_version(self) -> int:
        return self.policy_version

    def _optimizer_for_step(self) -> _Optimizer:
        return self.optimizer

    def policy_anchor_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self.anchor_state

    def load_policy_anchor_state_dict(self, state_dict) -> None:
        self.loaded_anchor_state = state_dict

    def reset_policy_anchor_to_current_model(self) -> None:
        self.reset_anchor_calls += 1


class _TrainingPaths:
    def __init__(self, root: Path) -> None:
        self.checkpoints_dir = root
        self.checkpoint_tracker_path = root / "checkpoint_tracker.json"
        self.logs_dir = root / "logs"
        self.latest_checkpoint_path = root / "latest.pt"
        self.best_checkpoint_path = root / "best.pt"
        self.snapshots_dir = root / "snapshots"
