from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import weiss_rl.training.loop.run_contexts as training_run_contexts


def test_training_run_context_builder_preserves_update_and_checkpoint_payloads(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=0)
    model = object()
    runtime = object()
    training_config = object()
    rewards_config = object()
    training_paths = object()
    replay_states = object()
    checkpoint_promotion_hooks = object()
    periodic_dev_eval_hooks = object()
    final_checkpoint_hooks = object()

    stack = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(system=SimpleNamespace(learner_torch_threads=7)),
    )
    setup = SimpleNamespace(
        learner=learner,
        model=model,
        runtime=runtime,
        training_config=training_config,
        rewards_config=rewards_config,
        training_paths=training_paths,
        algorithm="impala",
        latest_metrics={"setup": 1.0},
        init_schedule_offset_updates=11,
        resume_state={"checkpoint": "resume"},
        config_hash256="config-hash",
    )
    hooks = SimpleNamespace(
        central_runtime_actor_torch_threads=lambda received_stack, received_runtime: (
            events.append(("actor_threads", {"stack": received_stack, "runtime": received_runtime})) or 3
        ),
        apply_guidance_schedule_for_next_update=object(),
        entropy_coef_for_next_update=object(),
        collect_training_batch=object(),
        profile_block=object(),
        torch_num_threads_scope=object(),
        checkpoint_promotion=checkpoint_promotion_hooks,
        periodic_dev_eval=periodic_dev_eval_hooks,
        final_checkpoint=final_checkpoint_hooks,
    )

    def replay_states_from_config(received_training_config: object, *, repo_root: Path) -> object:
        events.append(("replay_states", {"training_config": received_training_config, "repo_root": repo_root}))
        return replay_states

    def reset_policy_anchor(**kwargs: object) -> None:
        events.append(("reset_anchor", kwargs))

    checkpoint_fn = object()
    dev_eval_fn = object()
    finalize_fn = object()
    contexts = training_run_contexts.build_training_run_contexts(
        stack=stack,
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        setup=setup,
        profile_timers=True,
        device=object(),
        checkpoint_interval_updates=5,
        run_id256="run-id",
        spec_hash256="spec-hash",
        tensorboard_logger=None,
        hooks=hooks,
        replay_states_from_config=replay_states_from_config,
        reset_policy_anchor=reset_policy_anchor,
        checkpoint_fn=checkpoint_fn,
        dev_eval_fn=dev_eval_fn,
        finalize_fn=finalize_fn,
    )

    assert [event[0] for event in events] == ["actor_threads", "replay_states", "reset_anchor"]
    assert events[0][1] == {"stack": stack, "runtime": runtime}
    assert events[1][1] == {"training_config": training_config, "repo_root": tmp_path}
    assert events[2][1] == {
        "learner": learner,
        "replay_states": replay_states,
        "resume_state": {"checkpoint": "resume"},
    }
    assert contexts.progress.latest_metrics is setup.latest_metrics
    assert contexts.update_inputs.learner is learner
    assert contexts.update_inputs.model is model
    assert contexts.update_inputs.runtime is runtime
    assert contexts.update_inputs.algorithm == "impala"
    assert contexts.update_inputs.training_config is training_config
    assert contexts.update_inputs.rewards_config is rewards_config
    assert contexts.update_inputs.replay_states is replay_states
    assert contexts.update_inputs.init_schedule_offset_updates == 11
    assert contexts.update_options.profile_timers is True
    assert contexts.update_options.actor_torch_threads == 3
    assert contexts.update_options.learner_torch_threads == 7
    assert contexts.update_hooks.collect_training_batch is hooks.collect_training_batch
    assert contexts.post_update_context.training_paths is training_paths
    assert contexts.post_update_context.config_hash256 == "config-hash"
    assert contexts.post_update_schedule.checkpoint_interval_updates == 5
    assert contexts.post_update_hooks.checkpoint_hooks is checkpoint_promotion_hooks
    assert contexts.post_update_hooks.periodic_dev_eval_hooks is periodic_dev_eval_hooks
    assert contexts.post_update_hooks.checkpoint_fn is checkpoint_fn
    assert contexts.post_update_hooks.dev_eval_fn is dev_eval_fn
    assert contexts.final_checkpoint_context.spec_hash256 == "spec-hash"
    assert contexts.final_checkpoint_hooks.hooks is final_checkpoint_hooks
    assert contexts.final_checkpoint_hooks.finalize_fn is finalize_fn
    assert contexts.actor_torch_threads == 3
    assert contexts.learner_torch_threads == 7
