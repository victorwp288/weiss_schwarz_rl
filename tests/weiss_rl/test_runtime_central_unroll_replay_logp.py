from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import weiss_rl.model as model_module
from weiss_rl.runtime import QueueRuntime, build_runtime_config


def test_central_structured_unroll_snapshots_replay_behavior_logp() -> None:
    pytest.importorskip("weiss_sim")
    from weiss_rl.config import apply_stack_overrides, load_stack_config, parse_override_tokens
    from weiss_rl.core.simulator_contract import load_verified_simulator_contract
    from weiss_rl.learners.action_logp import packed_scores_action_logp_and_entropy
    from weiss_rl.training.environments import spec_dimensions

    repo_root = Path(__file__).resolve().parents[2]
    stack = load_stack_config(repo_root / "configs" / "presets" / "typed_structured_v2.yaml")
    stack = apply_stack_overrides(
        stack,
        parse_override_tokens(
            [
                "system.actor_device=cpu",
                "system.learner_device=cpu",
                "system.collection_backend=central",
                "training.precision.mixed_precision=false",
            ]
        ),
    )
    contract = load_verified_simulator_contract(repo_root, expected_spec_hash="")
    observation_dim, action_dim = spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    device = torch.device("cpu")
    model = model_module.build_policy_value_model(
        observation_dim=observation_dim,
        config=stack.config.model,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    model.eval()
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=4,
        unroll_length=4,
        profile="fast",
        seed=20260513,
        pass_action_id=pass_action_id,
        runtime_mode="train_async_fast",
        minimal_batch=True,
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
        learner_device=device,
    )
    try:
        runtime._fill_pending_unrolls(target_count=int(runtime.config.batch_unrolls_per_update), occupancy_samples=[])
        unroll = runtime._select_pending_unrolls()[0]
        replay_model = cast(Any, runtime)._shared_actor_model or model
        replay_model.eval()
        obs = torch.as_tensor(unroll.obs, device=device, dtype=torch.float32)
        acting_seat = torch.as_tensor(unroll.to_play_seat, device=device, dtype=torch.long)
        initial_hidden = torch.as_tensor(unroll.initial_hidden_state, device=device, dtype=torch.float32)
        done = np.logical_or(unroll.terminated, unroll.truncated)
        reset_before_step = np.zeros_like(done, dtype=np.bool_)
        reset_before_step[1:] = done[:-1]
        actions = torch.as_tensor(unroll.actions, device=device, dtype=torch.long)
        behavior_logp = torch.as_tensor(unroll.behavior_logp, device=device, dtype=torch.float32)
        train_mask = torch.as_tensor(unroll.policy_train_mask, device=device, dtype=torch.bool)
        legal_actions = unroll.legal_actions
        assert legal_actions.ids is not None
        assert legal_actions.offsets is not None

        with torch.inference_mode():
            recurrent_flat, state_repr, observation_context, _values, _next_hidden = (
                replay_model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    initial_hidden,
                    reset_before_step=torch.as_tensor(reset_before_step, device=device, dtype=torch.bool),
                )
            )
            packed_scores = replay_model.score_packed_legal_candidates(
                recurrent_flat,
                obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
                legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
                scoring_mode="actor",
            )
            replay_logp, _entropy = packed_scores_action_logp_and_entropy(
                packed_scores,
                torch.as_tensor(legal_actions.ids, device=device, dtype=torch.long),
                torch.as_tensor(legal_actions.offsets, device=device, dtype=torch.long),
                actions,
                pass_action_id=pass_action_id,
            )

        delta = (replay_logp - behavior_logp).abs()
        assert float(delta[train_mask].max().item()) < 1e-5
    finally:
        runtime.close()
