from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset


def trajectory_bc_training_config(
    *,
    dataset_path: Path,
    batch_episodes: int = 4,
    aux_updates: int = 1,
    focus_source_labels: tuple[str, ...] = (),
    focus_fraction: float = 0.0,
    focus_groups: tuple[SimpleNamespace, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        structured_aux=SimpleNamespace(
            trajectory_bc_dataset_path=dataset_path.as_posix(),
            trajectory_bc_every_updates=1,
            trajectory_bc_aux_updates=aux_updates,
            trajectory_bc_batch_episodes=batch_episodes,
            trajectory_bc_seed=7,
            trajectory_bc_focus_source_labels=focus_source_labels,
            trajectory_bc_focus_fraction=focus_fraction,
            trajectory_bc_focus_groups=focus_groups,
        )
    )


class ReplayLearner:
    model = None
    teacher_aux_mode = "always"
    teacher_family_coef = 0.0
    teacher_slot_coef = 0.0
    teacher_hand_coef = 0.0
    teacher_move_source_coef = 0.0
    teacher_attack_type_coef = 0.0
    teacher_action_coef = 0.0
    teacher_same_family_action_coef = 0.0
    teacher_action_margin_coef = 0.0
    teacher_action_margin = 0.5
    teacher_same_family_action_margin_coef = 0.0
    teacher_same_family_action_margin = 0.5
    teacher_exact_action_families: tuple[str, ...] = ()
    teacher_public_heuristic_coef = 0.0
    teacher_public_heuristic_temperature = 32.0
    teacher_public_nonpass_over_pass_coef = 0.0
    teacher_public_nonpass_over_pass_margin = 0.5
    teacher_public_heuristic_families: tuple[str, ...] = ()
    teacher_public_heuristic_profiles: tuple[str, ...] = ()
    teacher_public_heuristic_profile_mode = ""
    teacher_public_heuristic_profiles_end_updates = -1

    def set_teacher_aux_coefs(self, **kwargs: object) -> None:
        attr_by_key = {
            "family": "teacher_family_coef",
            "slot": "teacher_slot_coef",
            "hand": "teacher_hand_coef",
            "move_source": "teacher_move_source_coef",
            "attack_type": "teacher_attack_type_coef",
            "action": "teacher_action_coef",
            "same_family_action": "teacher_same_family_action_coef",
            "action_margin": "teacher_action_margin_coef",
            "action_margin_value": "teacher_action_margin",
            "same_family_action_margin": "teacher_same_family_action_margin_coef",
            "same_family_action_margin_value": "teacher_same_family_action_margin",
            "exact_action_families": "teacher_exact_action_families",
            "public_heuristic": "teacher_public_heuristic_coef",
            "public_heuristic_temperature": "teacher_public_heuristic_temperature",
            "public_nonpass_over_pass": "teacher_public_nonpass_over_pass_coef",
            "public_nonpass_over_pass_margin": "teacher_public_nonpass_over_pass_margin",
            "public_heuristic_families": "teacher_public_heuristic_families",
            "public_heuristic_profiles": "teacher_public_heuristic_profiles",
            "public_heuristic_profile_mode": "teacher_public_heuristic_profile_mode",
            "public_heuristic_profiles_end_updates": "teacher_public_heuristic_profiles_end_updates",
        }
        for key, value in kwargs.items():
            setattr(self, attr_by_key[key], value)

    def auxiliary_update(self, batch: object) -> dict[str, float]:
        assert batch
        return {"teacher_action_acc": 1.0}


def dataset_with_source_labels(labels: list[str]) -> ReplayTrajectoryDataset:
    time_steps = 1
    episode_count = len(labels)
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int64)
    to_play_seat = np.zeros((time_steps, episode_count), dtype=np.int64)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    teacher = np.ones((time_steps, episode_count), dtype=np.int32)
    valid = np.ones((time_steps, episode_count), dtype=np.bool_)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    legal_action_meta = np.zeros((legal_ids.shape[0], 3), dtype=np.uint16)
    legal_offsets = np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32)
    metadata = {
        "format": "weiss_rl_replay_trajectory_bc_v1",
        "train_rows": episode_count,
        "selected_bundles": [
            {
                "source_dataset_label": label,
                "pair_index": index,
                "swap_index": 0,
                "outcome": "W",
            }
            for index, label in enumerate(labels)
        ],
    }
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        legal_action_meta=legal_action_meta,
        teacher_family=teacher,
        teacher_slot=teacher,
        teacher_move_source=teacher,
        teacher_attack_type=teacher,
        teacher_action=teacher,
        teacher_valid=valid,
        policy_train_mask=valid,
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata=metadata,
    )
