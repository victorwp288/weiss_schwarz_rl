from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from weiss_rl.masking import assert_strictly_increasing_legal_ids

weiss_sim: Any | None = None


@pytest.fixture(scope="module", autouse=True)
def _require_weiss_sim() -> None:
    global weiss_sim
    weiss_sim = pytest.importorskip(
        "weiss_sim",
        reason="simulator-backed rl smoke test requires weiss_sim on PYTHONPATH",
    )


def _sim() -> Any:
    assert weiss_sim is not None
    return weiss_sim


Layout = Literal["mask", "nomask", "i16_legal_ids"]
_LAYOUTS: tuple[Layout, ...] = ("mask", "nomask", "i16_legal_ids")
_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


def _make_pool(layout: Layout):
    kwargs = {"output_masks": False} if layout == "i16_legal_ids" else {}
    return _sim().make_pool(
        mode="train",
        num_envs=2,
        db_path=None,
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=123,
        layout=layout,
        **kwargs,
    )


def _assert_common_fields(step, *, num_envs: int) -> None:
    sim = _sim()
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


def _actions_from_mask(masks: np.ndarray, *, action_space: int) -> np.ndarray:
    sim = _sim()
    num_envs = int(masks.shape[0])
    assert masks.shape == (num_envs, action_space)
    actions = np.empty((num_envs,), dtype=np.uint32)
    for env_index in range(num_envs):
        legal_ids = np.flatnonzero(masks[env_index]).astype(np.uint32, copy=False)
        assert_strictly_increasing_legal_ids(legal_ids)
        actions[env_index] = sim.PASS_ACTION_ID if legal_ids.size == 0 else int(legal_ids[0])
    return actions


def _actions_from_packed_legal_ids(
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    *,
    num_envs: int,
    action_space: int,
) -> np.ndarray:
    sim = _sim()
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


def _assert_layout_contract(step, *, layout: Layout, buffers, action_space: int) -> np.ndarray:
    num_envs = int(step.obs.shape[0])
    _assert_common_fields(step, num_envs=num_envs)

    if layout == "mask":
        assert step.masks is not None
        assert step.legal_ids is None
        assert step.legal_offsets is None
        return _actions_from_mask(np.asarray(step.masks), action_space=action_space)

    assert step.masks is None
    if layout == "nomask":
        assert step.legal_ids is None
        assert step.legal_offsets is None
        legal_ids, legal_offsets = buffers.legal_action_ids()
        return _actions_from_packed_legal_ids(
            legal_ids,
            legal_offsets,
            num_envs=num_envs,
            action_space=action_space,
        )

    assert step.legal_ids is not None
    assert step.legal_offsets is not None
    return _actions_from_packed_legal_ids(
        np.asarray(step.legal_ids),
        np.asarray(step.legal_offsets),
        num_envs=num_envs,
        action_space=action_space,
    )


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_rl_step_contract_smoke_covers_supported_layouts(layout: Layout) -> None:
    sim = _sim()
    pool, buffers = _make_pool(layout)
    num_envs = int(pool.envs_len)
    action_space = int(pool.action_space)

    assert num_envs == 2
    assert action_space == int(sim.ACTION_SPACE_SIZE)

    reset_step = sim.rl.reset_rl(pool, layout=layout)
    actions = _assert_layout_contract(reset_step, layout=layout, buffers=buffers, action_space=action_space)
    assert actions.shape == (num_envs,)

    step_step = sim.rl.step_rl(pool, actions, layout=layout)
    next_actions = _assert_layout_contract(step_step, layout=layout, buffers=buffers, action_space=action_space)
    assert next_actions.shape == (num_envs,)
