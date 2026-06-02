from __future__ import annotations

from .test_impala_learner import (
    Any,
    ForwardProxyModel,
    ImpalaLearner,
    LegalActionBatch,
    Mapping,
    SequenceStructuredTeacherModel,
    SimpleNamespace,
    TinyPolicyValueModel,
    TinyStructuredTeacherModel,
    TrunkStructuredTeacherModel,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
    build_impala_forward_context,
    evaluate_impala_policy_forward,
    np,
    pytest,
    torch,
)


def test_build_impala_forward_context_detaches_outputs_and_checks_finiteness() -> None:
    calls: list[tuple[str, torch.Tensor, Any, dict[str, Any]]] = []
    learner = SimpleNamespace(
        _ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append((name, tensor, batch, context))
    )
    batch = {"forward_batch": True}
    logits = torch.ones((2, 1, 3), dtype=torch.float32, requires_grad=True)
    packed_logits = torch.arange(4, dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        values=values,
    )

    context = build_impala_forward_context(
        learner=learner,
        batch=batch,
        forward_result=forward_result,
    )

    torch.testing.assert_close(context["logits"], logits)
    torch.testing.assert_close(context["packed_logits"], packed_logits)
    torch.testing.assert_close(context["values"], values)
    assert context["logits"].requires_grad is False
    assert context["packed_logits"].requires_grad is False
    assert context["values"].requires_grad is False
    assert [
        (name, tensor, source_batch, call_context is context) for name, tensor, source_batch, call_context in calls
    ] == [
        ("forward_logits", logits, batch, True),
        ("forward_packed_logits", packed_logits, batch, True),
        ("forward_values", values, batch, True),
    ]


def test_build_impala_forward_context_skips_absent_logits_but_checks_values() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_ensure_finite_tensor=lambda name, tensor, *, batch, context: calls.append(name))
    values = torch.zeros((1, 1), dtype=torch.float32, requires_grad=True)

    context = build_impala_forward_context(
        learner=learner,
        batch={},
        forward_result=SimpleNamespace(logits=None, packed_logits=None, values=values),
    )

    assert context["logits"] is None
    assert context["packed_logits"] is None
    torch.testing.assert_close(context["values"], values)
    assert context["values"].requires_grad is False
    assert calls == ["forward_values"]


def test_evaluate_impala_policy_forward_uses_factorized_path_without_dense_forward() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1]], dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True]], dtype=torch.bool)
    original_packed = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)
    resolved_packed = (torch.as_tensor([1]), torch.as_tensor([0, 1]), None)
    factorized_result = SimpleNamespace(values=torch.as_tensor([[0.25], [0.5]], dtype=torch.float32))
    calls: list[tuple[str, Any]] = []

    def should_use_factorized(forward_model: object, *, packed_legal: object) -> bool:
        calls.append(("should_use_factorized", (forward_model, packed_legal)))
        return True

    def evaluate_factorized(
        source_batch: object,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        extra_active_mask: torch.Tensor | None,
    ) -> tuple[SimpleNamespace, tuple[torch.Tensor, torch.Tensor, None]]:
        calls.append(("evaluate_factorized", (source_batch, obs, actions, extra_active_mask)))
        return factorized_result, resolved_packed

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=should_use_factorized,
        _evaluate_factorized_time_major=evaluate_factorized,
    )
    forward_model = object()
    batch = object()

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=lambda _source, key: pytest.fail(f"unexpected batch_value({key})"),
        forward_model=forward_model,
        obs=obs,
        actions=actions,
        packed_legal=original_packed,
        loss_mask=loss_mask,
        reset_before_step=None,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    assert result.factorized_result is factorized_result
    assert result.packed_legal is resolved_packed
    assert result.logits is None
    assert result.packed_logits is None
    assert result.values is factorized_result.values
    assert result.forward_observation_context is None
    assert calls == [
        ("should_use_factorized", (forward_model, original_packed)),
        ("evaluate_factorized", (batch, obs, actions, retention_active)),
    ]


def test_evaluate_impala_policy_forward_forwards_dense_kwargs_and_restricts_rows() -> None:
    obs = torch.zeros((3, 1, 2), dtype=torch.float32)
    actions = torch.as_tensor([[0], [1], [0]], dtype=torch.long)
    loss_mask = torch.as_tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)
    retention_active = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    reset_before_step = torch.as_tensor([[False], [True], [False]], dtype=torch.bool)
    logits = torch.zeros((3, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((5,), dtype=torch.float32)
    values = torch.as_tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    observation_context = {"rows": obs.reshape(-1, obs.shape[-1])}
    batch = {
        "initial_hidden_state": "hidden",
        "to_play_seat": "seat",
        "actor": "actor",
        "legal_actions": "legal",
        "opponent_context_index": "opponent",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Mapping[str, object], key: str) -> object:
        calls.append(("batch_value", key))
        return source_batch[key]

    def forward_time_major(
        forward_obs: torch.Tensor,
        *,
        initial_hidden_state: object,
        to_play_seat: object,
        actor: object,
        legal_actions: object,
        policy_train_mask: torch.Tensor | None,
        reset_before_step: torch.Tensor | None,
        opponent_context_index: object,
    ) -> SimpleNamespace:
        calls.append(
            (
                "forward",
                (
                    forward_obs,
                    initial_hidden_state,
                    to_play_seat,
                    actor,
                    legal_actions,
                    policy_train_mask,
                    reset_before_step,
                    opponent_context_index,
                ),
            )
        )
        return SimpleNamespace(
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            observation_context=observation_context,
        )

    learner = SimpleNamespace(
        _should_use_factorized_legal_policy=lambda _forward_model, *, packed_legal: False,
        _forward_time_major=forward_time_major,
    )
    packed_legal = (torch.as_tensor([0, 1]), torch.as_tensor([0, 2]), None)

    result = evaluate_impala_policy_forward(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
        forward_model=object(),
        obs=obs,
        actions=actions,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
        reset_before_step=reset_before_step,
        trajectory_retention_active=retention_active,
        restrict_packed_policy_rows=True,
    )

    forward_call = calls[-1]
    assert forward_call[0] == "forward"
    forwarded_mask = forward_call[1][5]
    assert isinstance(forwarded_mask, torch.Tensor)
    assert forwarded_mask.dtype == loss_mask.dtype
    assert forwarded_mask.tolist() == [[1.0], [1.0], [0.0]]
    assert forward_call[1][0] is obs
    assert forward_call[1][1:5] == ("hidden", "seat", "actor", "legal")
    assert forward_call[1][6] is reset_before_step
    assert forward_call[1][7] == "opponent"
    assert calls[:5] == [
        ("batch_value", "initial_hidden_state"),
        ("batch_value", "to_play_seat"),
        ("batch_value", "actor"),
        ("batch_value", "legal_actions"),
        ("batch_value", "opponent_context_index"),
    ]
    assert result.factorized_result is None
    assert result.packed_legal is packed_legal
    assert result.logits is logits
    assert result.packed_logits is packed_logits
    assert result.values is values
    assert result.forward_observation_context is observation_context


def test_impala_learner_uses_compiled_forward_model_when_provided() -> None:
    base_model = TinyPolicyValueModel(action_dim=2)
    compiled_proxy = ForwardProxyModel(base_model)
    learner = ImpalaLearner(model=base_model, compiled_model=compiled_proxy)

    loss, _metrics = learner._loss_and_metrics(_simple_training_batch())

    assert float(loss.detach()) != 0.0
    assert compiled_proxy.forward_calls == 2


def test_impala_learner_forward_time_major_matches_manual_legacy_rollout() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=3)
    learner = ImpalaLearner(model=model)
    obs = torch.tensor(
        [
            [[0.25, -0.5], [1.0, 0.0]],
            [[-0.75, 0.5], [0.125, 0.25]],
        ],
        dtype=torch.float32,
    )
    initial_hidden = torch.ones((2, 1), dtype=torch.float32)

    with torch.no_grad():
        learner_logits, learner_values = learner._forward_time_major(obs, initial_hidden_state=initial_hidden)

        manual_hidden = initial_hidden
        manual_logits_steps: list[torch.Tensor] = []
        manual_value_steps: list[torch.Tensor] = []
        for step_obs in obs.unbind(dim=0):
            step_logits, step_value, manual_hidden = model(step_obs, manual_hidden)
            manual_logits_steps.append(step_logits)
            manual_value_steps.append(step_value)

    torch.testing.assert_close(learner_logits, torch.stack(manual_logits_steps, dim=0))
    torch.testing.assert_close(learner_values, torch.stack(manual_value_steps, dim=0))


def test_impala_learner_forward_time_major_requires_packed_meta_for_structured_updates() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    legal_actions = LegalActionBatch.from_packed(
        np.asarray([0, 5, 19], dtype=np.uint32),
        np.asarray([0, 3], dtype=np.uint32),
        action_space=action_catalog.action_space_size,
    )

    with pytest.raises(ValueError, match="packed legal_actions metadata"):
        learner._forward_time_major(
            obs,
            to_play_seat=np.asarray([[0]], dtype=np.int64),
            legal_actions=legal_actions,
        )


def test_impala_learner_forward_time_major_uses_sequence_fast_path_and_records_packed_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=SequenceStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, SequenceStructuredTeacherModel)
    assert logits.shape == (2, 1, action_catalog.action_space_size)
    assert values.shape == (2, 1)
    assert model.sequence_calls == 1
    assert model.step_calls == 0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(6.0)
    assert learner._active_timing_metrics["packed_candidate_rows"] == pytest.approx(2.0)
    assert learner._active_timing_metrics["avg_legal_actions_per_row"] == pytest.approx(3.0)
    assert learner._active_timing_metrics["timer_learner_forward_time_major_ms"] >= 0.0


def test_impala_learner_forward_time_major_uses_trunk_sequence_path_and_records_breakdown() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TrunkStructuredTeacherModel(action_catalog),
        profile_timers=True,
    )
    learner._active_timing_metrics = {}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    packed_ids = np.asarray([0, 5, 19, 1, 13, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    packed_logits, values = learner._forward_time_major(
        obs,
        to_play_seat=np.asarray([[0], [1]], dtype=np.int64),
        legal_actions=legal_actions,
    )

    model = learner.model
    assert isinstance(model, TrunkStructuredTeacherModel)
    assert packed_logits.shape == (6,)
    assert values.shape == (2, 1)
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.sequence_calls == 0
    assert learner._active_timing_metrics["timer_learner_trunk_ms"] >= 0.0
    assert learner._active_timing_metrics["timer_learner_packed_scorer_ms"] >= 0.0
