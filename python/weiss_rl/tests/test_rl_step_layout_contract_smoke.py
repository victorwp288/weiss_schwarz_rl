from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest

from weiss_rl.core.masking import assert_strictly_increasing_legal_ids
from weiss_rl.envs.decision_env import DecisionBoundaryEnv

MIN_WEISS_SIM_VERSION = (1, 2, 0)
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


Layout = Literal["mask", "nomask", "i16_legal_ids", "i16_legal_ids_nometa"]
_LAYOUTS: tuple[Layout, ...] = ("mask", "nomask", "i16_legal_ids", "i16_legal_ids_nometa")
_LEGAL_DECK = (list(range(1, 14)) * 4)[:50]


def _version_tuple(version: str) -> tuple[int, int, int]:
    release = version.strip().split("+", 1)[0].split("-", 1)[0]
    parts = [int(part) for part in release.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _make_pool(layout: Layout):
    kwargs = {"output_masks": False} if layout in {"i16_legal_ids", "i16_legal_ids_nometa"} else {}
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
    if layout == "i16_legal_ids_nometa":
        assert getattr(step, "legal_action_meta", None) is None
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


def test_weiss_sim_12_contract_surface_is_available() -> None:
    sim = _sim()
    version = getattr(sim, "__version__", "")

    assert _version_tuple(version) >= MIN_WEISS_SIM_VERSION
    assert int(sim.OBS_LEN) == 378
    assert int(sim.ACTION_SPACE_SIZE) == 527
    assert int(sim.SPEC_HASH) == 8590000130
    assert hasattr(sim, "make_pool")
    assert hasattr(sim, "EnvPoolBuffers")
    assert hasattr(sim, "export_spec_bundle")
    assert hasattr(sim.EnvPoolBuffers, "step_sample_from_logits_with_logp")
    assert hasattr(sim.EnvPoolBuffers, "legal_action_context_v1")

    bundle = sim.export_spec_bundle()
    assert int(bundle["observation"]["obs_len"]) == int(sim.OBS_LEN)
    assert int(bundle["action"]["action_space_size"]) == int(sim.ACTION_SPACE_SIZE)
    assert int(bundle["spec_hash"]) == int(sim.SPEC_HASH)


def test_env_pool_buffers_nometa_exposes_context_and_sampled_logp() -> None:
    sim = _sim()
    pool, buffers = sim.make_pool(
        mode="train",
        num_envs=2,
        db_path=None,
        deck_lists=[_LEGAL_DECK, _LEGAL_DECK],
        deck_ids=[101, 102],
        max_decisions=200,
        max_ticks=10_000,
        seed=456,
        profile="fast",
        layout="i16_legal_ids_nometa",
    )
    assert buffers.layout == "i16_legal_ids_nometa"

    reset_step = buffers.reset()
    assert reset_step.legal_ids is not None
    assert reset_step.legal_offsets is not None
    assert getattr(reset_step, "legal_action_meta", None) is None

    context, context_offsets = buffers.legal_action_context_v1()
    assert context.dtype == np.int32
    assert context_offsets.shape == reset_step.legal_offsets.shape
    assert int(context_offsets[0]) == 0
    assert int(context_offsets[-1]) == int(reset_step.legal_offsets[-1])
    assert context.shape[0] == int(context_offsets[-1])

    logits = np.zeros((int(pool.envs_len), int(sim.ACTION_SPACE_SIZE)), dtype=np.float32)
    seeds = np.array([11, 12], dtype=np.uint64)
    step, actions, action_logp = buffers.step_sample_from_logits_with_logp(logits, seeds)

    assert step.obs.shape == (int(pool.envs_len), int(sim.OBS_LEN))
    assert actions.shape == (int(pool.envs_len),)
    assert action_logp.shape == (int(pool.envs_len),)
    assert np.all(np.isfinite(action_logp))
    for env_index, action in enumerate(actions):
        start = int(reset_step.legal_offsets[env_index])
        end = int(reset_step.legal_offsets[env_index + 1])
        assert int(action) in set(int(value) for value in reset_step.legal_ids[start:end])


def test_decision_boundary_env_mask_and_ids_offsets_stay_step_equivalent() -> None:
    sim = _sim()
    num_envs = 4
    seed = 321

    mask_sim = sim.fast(
        num_envs=num_envs,
        seed=seed,
        max_decisions=200,
        max_ticks=10_000,
        observation_visibility="public",
        legal_repr="mask_u8",
        obs_dtype="i16",
    )
    ids_sim = sim.fast(
        num_envs=num_envs,
        seed=seed,
        max_decisions=200,
        max_ticks=10_000,
        observation_visibility="public",
        legal_repr="ids_u16",
        obs_dtype="i16",
    )

    mask_env = DecisionBoundaryEnv(mask_sim.pool, legality="mask", engine_status_policy="hard_fail")
    ids_env = DecisionBoundaryEnv(ids_sim.pool, legality="ids_offsets", engine_status_policy="hard_fail")

    try:
        mask_batch = mask_env.reset()
        ids_batch = ids_env.reset()

        for _ in range(8):
            assert np.array_equal(mask_batch.obs, ids_batch.obs)
            assert np.array_equal(mask_batch.reward, ids_batch.reward)
            assert np.array_equal(mask_batch.terminated, ids_batch.terminated)
            assert np.array_equal(mask_batch.truncated, ids_batch.truncated)
            assert np.array_equal(mask_batch.actor, ids_batch.actor)
            assert np.array_equal(mask_batch.decision_id, ids_batch.decision_id)
            assert np.array_equal(mask_batch.decision_kind, ids_batch.decision_kind)
            assert np.array_equal(mask_batch.engine_status, ids_batch.engine_status)
            assert np.array_equal(mask_batch.episode_seed, ids_batch.episode_seed)
            assert np.array_equal(mask_batch.episode_key, ids_batch.episode_key)
            assert np.array_equal(mask_batch.main_move_action, ids_batch.main_move_action)
            assert np.array_equal(mask_batch.main_pass_action, ids_batch.main_pass_action)

            mask_actions = _actions_from_mask(
                np.asarray(mask_batch.mask),
                action_space=int(sim.ACTION_SPACE_SIZE),
            )
            assert ids_batch.ids_offsets is not None
            ids_actions = _actions_from_packed_legal_ids(
                np.asarray(ids_batch.ids_offsets[0]),
                np.asarray(ids_batch.ids_offsets[1]),
                num_envs=num_envs,
                action_space=int(sim.ACTION_SPACE_SIZE),
            )
            assert np.array_equal(mask_actions, ids_actions)

            mask_batch = mask_env.step(mask_actions)
            ids_batch = ids_env.step(ids_actions)
    finally:
        mask_env.close()
        ids_env.close()
