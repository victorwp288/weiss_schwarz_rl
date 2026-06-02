from __future__ import annotations

from .test_impala_learner import (
    Any,
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    ImpalaLossBatchInputs,
    ImpalaLossForwardFlags,
    ImpalaLossMasks,
    ImpalaPolicyForwardResult,
    ImpalaTeacherTargetInputs,
    LegalActionBatch,
    SimpleNamespace,
    TinyPolicyValueModel,
    TrunkStructuredTeacherModel,
    VTraceTargets,
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
    assemble_impala_loss_inputs,
    cast,
    np,
    packed_scores_action_logp_and_entropy,
    packed_scores_family_entropy,
    prepare_impala_loss_inputs,
    pytest,
    resolve_impala_action_reductions,
    resolve_impala_dense_legal_mask,
    resolve_impala_loss_action_reductions,
    resolve_impala_loss_batch_inputs,
    resolve_impala_loss_forward_flags,
    resolve_impala_loss_masks,
    resolve_impala_loss_masks_stage,
    resolve_paired_auxiliary_batch_inputs,
    torch,
)


def test_impala_learner_packed_legal_actions_match_dense_mask_loss() -> None:
    torch.manual_seed(0)
    dense_model = TinyPolicyValueModel()
    packed_model = TinyPolicyValueModel()
    packed_model.load_state_dict(dense_model.state_dict())
    dense_learner = ImpalaLearner(model=dense_model, pass_action_id=2)
    packed_learner = ImpalaLearner(model=packed_model, pass_action_id=2)

    legal_mask = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    actions = np.asarray([[0], [2]], dtype=np.int64)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": actions,
        "legal_mask": legal_mask,
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask)
    packed_batch = dict(batch)
    packed_batch["legal_actions"] = LegalActionBatch.from_packed(packed_ids, packed_offsets)
    packed_batch["legal_mask"] = None

    dense_loss, dense_metrics = dense_learner._loss_and_metrics(batch)
    packed_loss, packed_metrics = packed_learner._loss_and_metrics(packed_batch)

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_batch["legal_mask"] is None
    assert dense_metrics == pytest.approx(packed_metrics)


def test_resolve_impala_action_reductions_uses_packed_candidate_family_entropy() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5], [12]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19, 10, 11, 12, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3, 7], dtype=torch.long)
    packed_meta = torch.as_tensor(
        _packed_meta_from_ids(action_catalog, packed_ids.numpy().astype(np.uint32, copy=False)),
        dtype=torch.long,
    )
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0, -1.0, 0.5, 3.0, 0.0], dtype=torch.float32)
    timings: list[tuple[str, float]] = []

    reductions = resolve_impala_action_reductions(
        factorized_result=None,
        logits=None,
        packed_logits=packed_logits,
        legal_mask=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        actions=actions,
        entropy_scope="family",
        pass_action_id=action_catalog.pass_action_id,
        action_catalog=action_catalog,
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )
    expected_logp, _candidate_entropy = packed_scores_action_logp_and_entropy(
        packed_logits,
        packed_ids,
        packed_offsets,
        actions,
        pass_action_id=action_catalog.pass_action_id,
    )
    expected_family_entropy = packed_scores_family_entropy(
        packed_logits,
        packed_offsets,
        packed_meta,
        row_shape=actions.shape,
        family_count=len(action_catalog.families),
    )

    torch.testing.assert_close(reductions.action_logp, expected_logp)
    torch.testing.assert_close(reductions.entropy, expected_family_entropy)
    assert reductions.action_logp.shape == actions.shape
    assert reductions.entropy.shape == actions.shape
    assert torch.isfinite(reductions.action_logp).all()
    assert torch.isfinite(reductions.entropy).all()
    assert [name for name, _duration in timings] == ["learner_packed_reductions"]
    assert timings[0][1] >= 0.0


def test_resolve_impala_action_reductions_preserves_family_entropy_requirements() -> None:
    action_catalog = _teacher_aux_catalog()
    actions = torch.as_tensor([[5]], dtype=torch.long)
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_logits = torch.as_tensor([0.0, 2.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="family entropy requires packed legal-action metadata and action_catalog"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=None,
            packed_logits=packed_logits,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )

    with pytest.raises(ValueError, match="family entropy requires packed candidate logits"):
        resolve_impala_action_reductions(
            factorized_result=None,
            logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
            packed_logits=None,
            legal_mask=None,
            packed_legal=(packed_ids, packed_offsets, None),
            actions=actions,
            entropy_scope="family",
            pass_action_id=action_catalog.pass_action_id,
            action_catalog=action_catalog,
            record_timing_ms=lambda _name, _duration: None,
        )


def test_resolve_impala_action_reductions_preserves_factorized_requirement_error() -> None:
    with pytest.raises(ValueError, match="factorized learner path requires action_logp and entropy"):
        resolve_impala_action_reductions(
            factorized_result=SimpleNamespace(action_logp=torch.zeros((1, 1)), entropy=None),
            logits=None,
            packed_logits=None,
            legal_mask=None,
            packed_legal=None,
            actions=torch.zeros((1, 1), dtype=torch.long),
            entropy_scope="candidate",
            pass_action_id=None,
            action_catalog=None,
            record_timing_ms=lambda _name, _duration: None,
        )


def test_resolve_impala_loss_batch_inputs_prefers_compiled_forward_model_and_expected_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    model = object()
    compiled_model = object()
    batch = {
        "vtrace_result": "vtrace",
        "obs": "raw_obs",
        "actions": "raw_actions",
    }
    calls: list[tuple[str, Any]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        calls.append(("batch_value", key))
        return source_batch.get(key)

    learner = SimpleNamespace(
        model=model,
        compiled_model=compiled_model,
        _require_obs=lambda value: calls.append(("require_obs", value)) or obs,
        _require_actions=lambda value, *, expected_shape: (
            calls.append(("require_actions", (value, expected_shape))) or actions
        ),
        _resolve_packed_legal_actions_with_meta=lambda source_batch, *, expected_shape: (
            calls.append(("packed_legal", (source_batch, expected_shape))) or packed_legal
        ),
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch=batch,
        batch_value=batch_value,
    )

    assert result.vtrace_result == "vtrace"
    assert result.obs is obs
    assert result.actions is actions
    assert result.packed_legal is packed_legal
    assert result.forward_model is compiled_model
    assert calls == [
        ("batch_value", "vtrace_result"),
        ("batch_value", "obs"),
        ("require_obs", "raw_obs"),
        ("batch_value", "actions"),
        ("require_actions", ("raw_actions", torch.Size((2, 1)))),
        ("packed_legal", (batch, torch.Size((2, 1)))),
    ]


def test_resolve_impala_loss_batch_inputs_falls_back_to_base_model() -> None:
    obs = torch.zeros((1, 1, 2), dtype=torch.float32)
    actions = torch.zeros((1, 1), dtype=torch.long)
    model = object()
    learner = SimpleNamespace(
        model=model,
        compiled_model=None,
        _require_obs=lambda _value: obs,
        _require_actions=lambda _value, *, expected_shape: actions,
        _resolve_packed_legal_actions_with_meta=lambda _batch, *, expected_shape: None,
    )

    result = resolve_impala_loss_batch_inputs(
        learner=learner,
        batch={"obs": object(), "actions": object()},
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert result.forward_model is model
    assert result.packed_legal is None


def test_assemble_impala_loss_inputs_preserves_stage_outputs_by_identity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    actions = torch.ones((2, 1), dtype=torch.long)
    loss_mask = torch.ones((2, 1), dtype=torch.float32)
    reset_before_step = torch.zeros((2, 1), dtype=torch.bool)
    trajectory_retention_valid = torch.ones((2, 1), dtype=torch.float32)
    logits = torch.zeros((2, 1, 4), dtype=torch.float32)
    packed_logits = torch.zeros((3,), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    public_target_logits = torch.full((3,), 0.25, dtype=torch.float32)
    original_packed_legal = (torch.as_tensor([0]), torch.as_tensor([0, 1]), None)
    forward_packed_legal = (torch.as_tensor([1, 2, 3]), torch.as_tensor([0, 3]), None)
    forward_model = object()
    factorized_result = object()
    observation_context = {"obs": obs.reshape(-1, obs.shape[-1])}
    context = {"logits": logits, "values": values}
    packed_view = cast(Any, object())
    teacher_aux_packed_view = cast(Any, object())

    assembled = assemble_impala_loss_inputs(
        batch_inputs=ImpalaLossBatchInputs(
            vtrace_result="vtrace",
            obs=obs,
            actions=actions,
            packed_legal=original_packed_legal,
            forward_model=forward_model,
        ),
        masks=ImpalaLossMasks(
            loss_mask=loss_mask,
            reset_before_step=reset_before_step,
            trajectory_retention_valid=trajectory_retention_valid,
            trajectory_retention_active=None,
        ),
        forward_flags=ImpalaLossForwardFlags(
            teacher_aux_active=True,
            emit_structured_metrics=True,
            restrict_packed_policy_rows=False,
        ),
        forward_result=ImpalaPolicyForwardResult(
            factorized_result=factorized_result,
            packed_legal=forward_packed_legal,
            logits=logits,
            packed_logits=packed_logits,
            values=values,
            forward_observation_context=observation_context,
        ),
        legal_mask=legal_mask,
        teacher_target_inputs=ImpalaTeacherTargetInputs(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_aux_packed_view,
            public_heuristic_target_logits=public_target_logits,
        ),
        context=context,
    )

    assert assembled.vtrace_result == "vtrace"
    assert assembled.obs is obs
    assert assembled.actions is actions
    assert assembled.packed_legal is forward_packed_legal
    assert assembled.packed_legal is not original_packed_legal
    assert assembled.forward_model is forward_model
    assert assembled.loss_mask is loss_mask
    assert assembled.reset_before_step is reset_before_step
    assert assembled.trajectory_retention_valid is trajectory_retention_valid
    assert assembled.teacher_aux_active is True
    assert assembled.emit_structured_metrics is True
    assert assembled.factorized_result is factorized_result
    assert assembled.logits is logits
    assert assembled.packed_logits is packed_logits
    assert assembled.values is values
    assert assembled.forward_observation_context is observation_context
    assert assembled.legal_mask is legal_mask
    assert assembled.packed_view is packed_view
    assert assembled.teacher_aux_packed_view is teacher_aux_packed_view
    assert assembled.public_heuristic_target_logits is public_target_logits
    assert assembled.context is context


def test_prepare_impala_loss_inputs_restricts_packed_forward_to_policy_and_retention_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.4,
    )
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }
    captured_policy_masks: list[torch.Tensor | None] = []

    def fake_forward(
        obs: torch.Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: Any = None,
        policy_train_mask: torch.Tensor | None = None,
        reset_before_step: torch.Tensor | None = None,
        opponent_context_index: Any = None,
    ) -> SimpleNamespace:
        del initial_hidden_state, to_play_seat, actor, legal_actions, reset_before_step, opponent_context_index
        captured_policy_masks.append(None if policy_train_mask is None else policy_train_mask.detach().clone())
        return SimpleNamespace(
            logits=None,
            packed_logits=torch.zeros((int(packed_ids.shape[0]),), dtype=torch.float32),
            values=torch.zeros(obs.shape[:2], dtype=torch.float32),
            observation_context={"rows": obs.reshape(-1, obs.shape[-1])},
        )

    cast(Any, learner)._forward_time_major = fake_forward

    prepared = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))

    assert prepared.packed_legal is not None
    assert prepared.legal_mask is None
    assert prepared.teacher_aux_active is False
    assert prepared.emit_structured_metrics is False
    assert captured_policy_masks
    assert captured_policy_masks[0] is not None
    assert captured_policy_masks[0].tolist() == [[1.0], [1.0]]
    assert prepared.context["packed_logits"].shape == (int(packed_ids.shape[0]),)
    assert prepared.context["values"].tolist() == [[0.0], [0.0]]


def test_resolve_impala_loss_masks_converts_reset_and_retention_activity() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), trajectory_retention_coef=0.4)
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    batch = {
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "reset_before_step": np.asarray([[False], [True]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }

    masks = resolve_impala_loss_masks(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.tolist() == [[1.0], [0.0]]
    assert masks.reset_before_step is not None
    assert masks.reset_before_step.dtype == torch.bool
    assert masks.reset_before_step.tolist() == [[False], [True]]
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[0.0], [1.0]]
    assert masks.trajectory_retention_active is not None
    assert masks.trajectory_retention_active.dtype == torch.bool
    assert masks.trajectory_retention_active.tolist() == [[False], [True]]


def test_resolve_impala_loss_masks_defaults_policy_mask_and_disables_retention_activity() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float64)
    batch = {
        "trajectory_retention_valid": np.asarray([[True], [False]], dtype=np.bool_),
    }
    calls: list[tuple[str, Any]] = []
    learner = SimpleNamespace(
        trajectory_retention_coef=0.0,
        _optional_time_major_loss_mask=lambda value, *, expected_shape, like: (
            calls.append(("mask", (value, expected_shape, like.shape, like.dtype)))
            or (torch.as_tensor(value, dtype=torch.float32) if value is not None else None)
        ),
    )

    masks = resolve_impala_loss_masks_stage(
        learner=learner,
        batch=batch,
        obs=obs,
        batch_value=lambda source, key: source.get(key),
    )

    assert masks.loss_mask.dtype == obs.dtype
    assert masks.loss_mask.device == obs.device
    assert masks.loss_mask.tolist() == [[1.0], [1.0]]
    assert masks.reset_before_step is None
    assert masks.trajectory_retention_valid is not None
    assert masks.trajectory_retention_valid.tolist() == [[1.0], [0.0]]
    assert masks.trajectory_retention_active is None
    assert calls[0] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[1] == ("mask", (None, torch.Size((2, 1)), torch.Size((2, 1)), torch.float64))
    assert calls[2][0] == "mask"
    assert calls[2][1][0] is batch["trajectory_retention_valid"]
    assert calls[2][1][1:] == (torch.Size((2, 1)), torch.Size((2, 1)), torch.float64)


def test_resolve_impala_loss_forward_flags_only_restricts_safe_packed_policy_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_legal = (
        torch.as_tensor([0, action_catalog.pass_action_id], dtype=torch.long),
        torch.as_tensor([0, 2], dtype=torch.long),
        torch.as_tensor(_packed_meta_from_ids(action_catalog, np.asarray([0, action_catalog.pass_action_id]))),
    )
    loss_mask = torch.as_tensor([[1.0], [0.0]], dtype=torch.float32)

    plain = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
    )
    teacher_model = TinyPolicyValueModel(action_dim=action_catalog.action_space_size)
    teacher_model.action_catalog = action_catalog
    teacher = ImpalaLearner(model=teacher_model, teacher_action_coef=0.5, structured_metrics_mode="off")
    structured = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="full",
    )

    plain_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=packed_legal, loss_mask=loss_mask)
    teacher_flags = resolve_impala_loss_forward_flags(learner=teacher, packed_legal=packed_legal, loss_mask=loss_mask)
    structured_flags = resolve_impala_loss_forward_flags(
        learner=structured,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
    )
    dense_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=None, loss_mask=loss_mask)

    assert plain_flags.teacher_aux_active is False
    assert plain_flags.emit_structured_metrics is False
    assert plain_flags.restrict_packed_policy_rows is True
    assert teacher_flags.teacher_aux_active is True
    assert teacher_flags.restrict_packed_policy_rows is False
    assert structured_flags.emit_structured_metrics is True
    assert structured_flags.restrict_packed_policy_rows is False
    assert dense_flags.restrict_packed_policy_rows is False


def test_resolve_impala_dense_legal_mask_returns_none_for_packed_legal_without_resolving() -> None:
    calls: list[str] = []
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: calls.append("resolve"))
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        None,
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch={},
        obs=obs,
        packed_legal=packed_legal,
        logits=logits,
    )

    assert result is None
    assert calls == []


def test_resolve_impala_dense_legal_mask_resolves_and_validates_dense_shape() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones_like(logits, dtype=torch.bool)
    batch = {"dense": True}
    calls: list[tuple[Any, torch.Size, int]] = []
    learner = SimpleNamespace(
        _resolve_legal_mask=lambda source_batch, *, expected_shape, action_dim: (
            calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        )
    )

    result = resolve_impala_dense_legal_mask(
        learner=learner,
        batch=batch,
        obs=obs,
        packed_legal=None,
        logits=logits,
    )

    assert result is legal_mask
    assert calls == [(batch, torch.Size((2, 1)), 5)]


def test_resolve_impala_dense_legal_mask_rejects_missing_logits_and_shape_mismatch() -> None:
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    learner = SimpleNamespace(_resolve_legal_mask=lambda *args, **kwargs: torch.ones((2, 1, 4), dtype=torch.bool))

    with pytest.raises(ValueError, match="dense learner path requires dense logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=None,
        )
    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        resolve_impala_dense_legal_mask(
            learner=learner,
            batch={},
            obs=obs,
            packed_legal=None,
            logits=logits,
        )


def test_prepare_impala_loss_inputs_rejects_dense_legal_mask_shape_mismatch() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
    }

    def bad_legal_mask(_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        del expected_shape, action_dim
        return torch.ones((1, 1, 2), dtype=torch.bool)

    cast(Any, learner)._resolve_legal_mask = bad_legal_mask

    with pytest.raises(ValueError, match="legal_mask must match learner logits"):
        prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))


def test_resolve_impala_loss_action_reductions_attaches_detached_context_and_checks_finiteness() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()
    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    finite_calls: list[str] = []

    def record_finite(name: str, tensor: torch.Tensor, *, batch: Any, context: dict[str, Any]) -> None:
        del tensor, batch, context
        finite_calls.append(name)

    cast(Any, learner)._ensure_finite_tensor = record_finite

    reductions = resolve_impala_loss_action_reductions(
        learner=learner,
        batch=batch,
        loss_inputs=inputs,
    )

    assert reductions.context is inputs.context
    assert reductions.context["action_logp"].shape == torch.Size((2, 1))
    assert reductions.context["entropy"].shape == torch.Size((2, 1))
    assert reductions.context["action_logp"].requires_grad is False
    assert reductions.context["entropy"].requires_grad is False
    assert finite_calls == ["action_logp", "entropy"]


def test_resolve_paired_auxiliary_batch_inputs_preserves_default_loss_mask_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(model=TinyPolicyValueModel(), pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "legal_ids": np.concatenate([packed_ids, packed_ids]),
        "legal_offsets": np.asarray([0, 3, 6], dtype=np.uint32),
        "legal_action_meta": _packed_meta_from_ids(action_catalog, np.concatenate([packed_ids, packed_ids])),
    }

    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired helper requires packed legal actions",
    )

    assert inputs.obs.shape == (2, 1, 2)
    assert inputs.expected_shape == torch.Size([2, 1])
    assert inputs.loss_mask.shape == torch.Size([2, 1])
    assert torch.all(inputs.loss_mask == 1.0)
    assert inputs.packed_legal[0].tolist() == [0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id]


def test_resolve_paired_auxiliary_batch_inputs_preserves_missing_packed_error() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    batch = {"obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32)}

    with pytest.raises(ValueError, match="paired helper requires packed legal actions"):
        resolve_paired_auxiliary_batch_inputs(
            learner,
            batch,
            packed_legal_error="paired helper requires packed legal actions",
        )


def test_impala_learner_dense_trajectory_retention_is_separate_from_policy_train_mask() -> None:
    torch.manual_seed(0)
    base_model = TinyPolicyValueModel(action_dim=2)
    retention_model = TinyPolicyValueModel(action_dim=2)
    retention_model.load_state_dict(base_model.state_dict())
    base_learner = ImpalaLearner(model=base_model)
    retention_learner = ImpalaLearner(model=retention_model, trajectory_retention_coef=0.4)
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    base_loss, _base_metrics = base_learner._loss_and_metrics(batch)
    retention_loss, retention_metrics = retention_learner._loss_and_metrics(batch)

    assert retention_metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert retention_metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_supported_fraction"] == pytest.approx(1.0)
    assert retention_metrics["trajectory_retention_weighted_loss"] > 0.0
    assert float(retention_loss.detach()) == pytest.approx(
        float(base_loss.detach()) + retention_metrics["trajectory_retention_weighted_loss"]
    )


def test_impala_learner_uses_factorized_legal_policy_path_for_loss_and_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        profile_timers=True,
        trajectory_retention_coef=0.06,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [11]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.1], [0.2]], dtype=np.float32),
            pg_advantages=np.asarray([[1.0], [0.5]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    loss, metrics = learner._loss_and_metrics(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert float(loss.detach()) != 0.0
    assert model.factorized_calls == 1
    assert learner._active_timing_metrics["timer_learner_factorized_policy_ms"] >= 0.0
    assert learner._active_timing_metrics["packed_candidate_count"] == pytest.approx(7.0)
    assert metrics["entropy"] > 0.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["trajectory_retention_loss"] == pytest.approx(0.25)
    assert metrics["trajectory_retention_weighted_loss"] == pytest.approx(0.015)


def test_impala_learner_restricts_packed_policy_scoring_to_train_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[5], [11]], dtype=np.int64),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=packed_meta,
            action_space=action_catalog.action_space_size,
        ),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": np.asarray([[-2.0], [-3.0]], dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert float(loss.detach()) != 0.0
    assert model.trunk_calls == 1
    assert model.scorer_calls == 1
    assert model.scorer_row_count == 1
    assert model.scorer_candidate_count == 3
    assert learner._active_timing_metrics["packed_candidate_train_rows"] == pytest.approx(1.0)
    assert learner._active_timing_metrics["packed_candidate_train_count"] == pytest.approx(3.0)
    assert float(context["vtrace_rhos"][1, 0]) == pytest.approx(1.0)
    assert float(context["vtrace_rhos"][0, 0]) > 1.0
    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
