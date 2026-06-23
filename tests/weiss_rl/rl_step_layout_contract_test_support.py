from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from weiss_rl.core.masking import assert_strictly_increasing_legal_ids

MIN_WEISS_SIM_VERSION = (1, 2, 0)
Layout = Literal["mask", "nomask", "i16_legal_ids", "i16_legal_ids_nometa"]
LAYOUTS: tuple[Layout, ...] = ("mask", "nomask", "i16_legal_ids", "i16_legal_ids_nometa")
LEGAL_DECK = (list(range(1, 14)) * 4)[:50]

_weiss_sim: Any | None = None


def sim_module() -> Any:
    global _weiss_sim
    if _weiss_sim is None:
        _weiss_sim = pytest.importorskip(
            "weiss_sim",
            reason="simulator-backed rl smoke test requires weiss_sim on PYTHONPATH",
        )
    return _weiss_sim


def version_tuple(version: str) -> tuple[int, int, int]:
    release = version.strip().split("+", 1)[0].split("-", 1)[0]
    parts = [int(part) for part in release.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def make_pool(layout: Layout):
    kwargs = {"output_masks": False} if layout in {"i16_legal_ids", "i16_legal_ids_nometa"} else {}
    return sim_module().make_pool(
        mode="train",
        num_envs=2,
        db_path=None,
        deck_lists=[LEGAL_DECK, LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=123,
        layout=layout,
        **kwargs,
    )


def actions_from_mask(masks: np.ndarray, *, action_space: int) -> np.ndarray:
    sim = sim_module()
    num_envs = int(masks.shape[0])
    assert masks.shape == (num_envs, action_space)
    actions = np.empty((num_envs,), dtype=np.uint32)
    for env_index in range(num_envs):
        legal_ids = np.flatnonzero(masks[env_index]).astype(np.uint32, copy=False)
        assert_strictly_increasing_legal_ids(legal_ids)
        actions[env_index] = sim.PASS_ACTION_ID if legal_ids.size == 0 else int(legal_ids[0])
    return actions


def actions_from_packed_legal_ids(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    num_envs: int,
    action_space: int,
) -> np.ndarray:
    sim = sim_module()
    assert legal_offsets.shape == (num_envs + 1,)
    assert int(legal_offsets[0]) == 0
    assert np.all(legal_offsets[1:] >= legal_offsets[:-1])

    used = int(legal_offsets[-1])
    assert 0 <= used <= int(legal_ids.shape[0])
    used_legal_ids = np.asarray(legal_ids[:used], dtype=np.uint32)

    actions = np.empty((num_envs,), dtype=np.uint32)
    for env_index in range(num_envs):
        start = int(legal_offsets[env_index])
        end = int(legal_offsets[env_index + 1])
        env_legal_ids = used_legal_ids[start:end]
        assert_strictly_increasing_legal_ids(env_legal_ids)
        assert np.all(env_legal_ids < action_space)
        actions[env_index] = sim.PASS_ACTION_ID if start == end else int(env_legal_ids[0])
    return actions


def assert_common_fields(step, *, num_envs: int) -> None:
    sim = sim_module()
    assert step.obs.shape == (num_envs, int(sim.OBS_LEN))
    assert step.rewards.shape == (num_envs,)
    assert step.terminated.shape == (num_envs,)
    assert step.truncated.shape == (num_envs,)
    assert step.actor.shape == (num_envs,)
    assert step.decision_kind.shape == (num_envs,)
    assert step.decision_id.shape == (num_envs,)
    assert step.engine_status.shape == (num_envs,)
    assert step.spec_hash.shape == (num_envs,)
    assert np.array_equal(step.spec_hash, np.full((num_envs,), sim.SPEC_HASH, dtype=step.spec_hash.dtype))


def assert_layout_contract(step, *, layout: Layout, buffers, action_space: int) -> np.ndarray:
    num_envs = int(step.obs.shape[0])
    assert_common_fields(step, num_envs=num_envs)

    if layout == "mask":
        assert step.masks is not None
        assert step.legal_ids is None
        assert step.legal_offsets is None
        return actions_from_mask(np.asarray(step.masks), action_space=action_space)

    assert step.masks is None
    if layout == "nomask":
        assert step.legal_ids is None
        assert step.legal_offsets is None
        legal_ids, legal_offsets = buffers.legal_action_ids()
        return actions_from_packed_legal_ids(
            legal_ids,
            legal_offsets,
            num_envs=num_envs,
            action_space=action_space,
        )

    assert step.legal_ids is not None
    assert step.legal_offsets is not None
    if layout == "i16_legal_ids_nometa":
        assert getattr(step, "legal_action_meta", None) is None
    return actions_from_packed_legal_ids(
        np.asarray(step.legal_ids),
        np.asarray(step.legal_offsets),
        num_envs=num_envs,
        action_space=action_space,
    )
