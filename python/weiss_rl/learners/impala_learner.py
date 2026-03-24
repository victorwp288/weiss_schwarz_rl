"""IMPALA learner scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


@dataclass(slots=True)
class ImpalaLearner:
    learning_rate: float = 2e-4
    checkpoint_dir: Path | None = None
    checkpoint_interval_updates: int = 50000

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)

    def update(self, batch: Any) -> dict[str, float]:
        """Learner update hook."""
        _ = batch
        self.update_count += 1

        if self.checkpoint_dir and self.update_count % self.checkpoint_interval_updates == 0:
            self.policy_version += 1
            self._save_checkpoint()

        return {"loss": 0.0}

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.update_count}.pt"
        checkpoint_path.write_text(
            f"update_count: {self.update_count}\npolicy_version: {self.policy_version}\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def get_policy_version(self) -> int:
        return self.policy_version
