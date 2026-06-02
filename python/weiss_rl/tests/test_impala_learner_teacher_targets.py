from __future__ import annotations

from .test_impala_learner import (
    Any,
    ImpalaLearner,
    SimpleNamespace,
    TinyPolicyValueModel,
    TinyStructuredTeacherModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
    _TeacherTargetInputLearner,
    apply_impala_teacher_auxiliary,
    apply_impala_teacher_auxiliary_stage,
    cast,
    compute_impala_teacher_auxiliary,
    impala_loss_teacher_stage,
    impala_loss_teacher_targets_stage,
    np,
    prepare_impala_loss_teacher_target_inputs,
    prepare_impala_teacher_target_inputs,
    pytest,
    resolve_impala_teacher_auxiliary_coefficients,
    resolve_impala_teacher_auxiliary_factorized_inputs,
    resolve_impala_teacher_auxiliary_inputs,
    resolve_impala_teacher_auxiliary_labels,
    resolve_impala_teacher_auxiliary_packed_inputs,
    resolve_impala_teacher_target_plan,
    torch,
)


def test_prepare_impala_teacher_target_inputs_builds_packed_view_and_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)
    obs = torch.ones((1, 1, 2), dtype=torch.float32)
    packed_logits = torch.as_tensor([0.0, 1.0, -1.0], dtype=torch.float32)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=obs,
        logits=None,
        packed_logits=packed_logits,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context={"obs": obs.reshape(1, 2)},
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.as_tensor([0.0, 1.0, 2.0]))
    assert learner.packed_public_target_calls == 1
    assert [name for name, _duration in learner.timings] == ["learner_packed_view", "learner_public_heuristic_target"]


def test_prepare_impala_teacher_target_inputs_respects_teacher_aux_gate() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=torch.zeros((1, 1, _teacher_aux_catalog().action_space_size), dtype=torch.float32),
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=False,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is None
    assert learner.packed_public_target_calls == 0
    assert [name for name, _duration in learner.timings] == ["learner_packed_view"]


def test_resolve_impala_teacher_target_plan_names_candidate_target_gates() -> None:
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0

    disabled = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=False,
    )
    unsupported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )
    supported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )

    assert disabled.public_candidate_target_active is True
    assert disabled.factorized_candidate_teacher_view_active is False
    assert disabled.can_prepare_candidate_targets is False
    assert unsupported.can_prepare_candidate_targets is False
    assert supported.can_prepare_candidate_targets is True


def test_resolve_impala_teacher_target_plan_allows_factorized_margin_without_public_model_support() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )

    plan = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        teacher_aux_enabled=True,
    )

    assert plan.public_candidate_target_active is False
    assert plan.factorized_candidate_teacher_view_active is True
    assert plan.can_prepare_candidate_targets is True


def test_prepare_impala_teacher_target_inputs_scores_factorized_public_target_when_active() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.ones((3,), dtype=torch.float32))
    assert learner.factorized_teacher_view_calls == [True]


def test_prepare_impala_teacher_target_inputs_requests_factorized_margin_view_without_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is None
    assert learner.factorized_teacher_view_calls == [False]
    assert learner.timings == []


def test_apply_impala_teacher_auxiliary_returns_unchanged_loss_when_inactive() -> None:
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    total_loss = torch.tensor(2.0)

    result = apply_impala_teacher_auxiliary(
        learner=object(),
        batch={},
        total_loss=total_loss,
        context=context,
        teacher_aux_active=False,
        logits=None,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=None,
        expected_shape=torch.Size((1, 1)),
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda _batch, _shape, _action_dim: pytest.fail(
            "inactive teacher aux must not resolve mask"
        ),
        batch_value=lambda batch, key: getattr(batch, key),
    )

    assert result.total_loss is total_loss
    assert result.teacher_metrics == {}
    assert list(context) == ["existing"]
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))


def test_apply_impala_teacher_auxiliary_resolves_dense_mask_for_packed_without_meta() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -1.0
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }
    context: dict[str, Any] = {}
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    result = apply_impala_teacher_auxiliary(
        learner=learner,
        batch=batch,
        total_loss=torch.tensor(1.0),
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=torch.Size((1, 1)),
        packed_legal=(packed_ids, packed_offsets, None),
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
            resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert resolver_calls == [(batch, torch.Size((1, 1)), action_catalog.action_space_size)]
    assert result.total_loss.item() > 1.0
    assert result.teacher_metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert "teacher_aux_loss" in context


def test_apply_impala_teacher_auxiliary_stage_maps_loss_inputs_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(2.0, dtype=torch.float32)
    resolved_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    learner = SimpleNamespace(_resolve_legal_mask=resolve_legal_mask)
    batch = {"teacher_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    logits = torch.zeros((2, 1, 7), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 7), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    public_targets = torch.zeros((2, 1, 7), dtype=torch.float32)
    values = torch.zeros((2, 1), dtype=torch.float32)
    inputs = SimpleNamespace(
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        values=values,
        packed_legal=packed_legal,
        teacher_aux_packed_view=packed_view,
        factorized_result=factorized_result,
        public_heuristic_target_logits=public_targets,
    )
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_apply_impala_teacher_auxiliary(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        assert kwargs["resolve_legal_mask"](batch, torch.Size((2, 1)), 7) is resolved_mask
        return SimpleNamespace(
            total_loss=kwargs["total_loss"] + torch.tensor(0.25, dtype=torch.float32),
            teacher_metrics={"teacher_valid_fraction": 0.5},
        )

    monkeypatch.setattr(
        impala_loss_teacher_stage,
        "apply_impala_teacher_auxiliary",
        fake_apply_impala_teacher_auxiliary,
    )

    result = apply_impala_teacher_auxiliary_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["total_loss"] is total_loss
    assert captured["context"] is context
    assert captured["teacher_aux_active"] is True
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] == "catalog"
    assert captured["expected_shape"] == values.shape
    assert captured["packed_legal"] is packed_legal
    assert captured["packed_view"] is packed_view
    assert captured["factorized_result"] is factorized_result
    assert captured["public_heuristic_target_logits"] is public_targets
    assert captured["batch_value"] is batch_value
    assert resolver_calls == [(batch, torch.Size((2, 1)), 7)]
    assert batch_values == []
    torch.testing.assert_close(result.total_loss, torch.tensor(2.25))
    assert result.teacher_metrics == {"teacher_valid_fraction": 0.5}


def test_prepare_impala_loss_teacher_target_inputs_maps_forward_state_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"teacher_target_batch": True}
    forward_model = object()
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_logits = torch.arange(4, dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_observation_context = {"encoded": torch.ones((2, 1), dtype=torch.float32)}
    masks = SimpleNamespace(loss_mask=loss_mask)
    forward_flags = SimpleNamespace(emit_structured_metrics=False, teacher_aux_active=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_observation_context=forward_observation_context,
    )
    packed_view = object()
    teacher_view = object()
    public_targets = torch.ones((4,), dtype=torch.float32)
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_view,
            public_heuristic_target_logits=public_targets,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    result = prepare_impala_loss_teacher_target_inputs(
        learner=learner,
        batch=batch,
        forward_model=forward_model,
        obs=obs,
        masks=masks,
        forward_flags=forward_flags,
        forward_result=forward_result,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["forward_model"] is forward_model
    assert captured["obs"] is obs
    assert captured["logits"] is logits
    assert captured["packed_logits"] is packed_logits
    assert captured["packed_legal"] is packed_legal
    assert captured["loss_mask"] is loss_mask
    assert captured["factorized_result"] is factorized_result
    assert captured["forward_observation_context"] is forward_observation_context
    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is True
    assert result.packed_view is packed_view
    assert result.teacher_aux_packed_view is teacher_view
    assert result.public_heuristic_target_logits is public_targets


def test_prepare_impala_loss_teacher_target_inputs_needs_packed_view_for_structured_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=None,
            teacher_aux_packed_view=None,
            public_heuristic_target_logits=None,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    prepare_impala_loss_teacher_target_inputs(
        learner=object(),
        batch={},
        forward_model=object(),
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        masks=SimpleNamespace(loss_mask=torch.ones((1, 1), dtype=torch.float32)),
        forward_flags=SimpleNamespace(emit_structured_metrics=True, teacher_aux_active=False),
        forward_result=SimpleNamespace(
            logits=None,
            packed_logits=None,
            packed_legal=None,
            factorized_result=None,
            forward_observation_context=None,
        ),
    )

    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is False


def test_compute_impala_teacher_auxiliary_request_preserves_dense_teacher_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
        profile_timers=True,
    )
    cast(Any, learner)._active_timing_metrics = {}
    expected_shape = torch.Size((1, 1))
    result = compute_impala_teacher_auxiliary(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        logits=torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32),
        legal_mask=torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool),
        loss_mask=torch.ones(expected_shape, dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=expected_shape,
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        batch_value=lambda batch, key: batch.get(key),
    )

    assert result.loss > 0.0
    assert result.metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.metrics["teacher_family_loss"] > 0.0
    assert result.metrics["teacher_action_loss"] > 0.0
    assert "teacher_family_log_probs" in result.context
    assert cast(Any, learner)._active_timing_metrics["timer_learner_teacher_aux_ms"] >= 0.0


def test_resolve_impala_teacher_auxiliary_labels_preserves_time_major_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(model=TinyStructuredTeacherModel(action_catalog))
    expected_shape = torch.Size((1, 2))
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"], family_index["attack"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0, 1]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1, 0]], dtype=np.int64),
        "teacher_action": np.asarray([[0, 11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True, False]], dtype=np.bool_),
    }

    labels = resolve_impala_teacher_auxiliary_labels(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=expected_shape,
    )

    assert labels.family is not None
    assert labels.family.dtype == torch.long
    assert labels.family.shape == expected_shape
    assert labels.family.tolist() == [[family_index["main_play_character"], family_index["attack"]]]
    assert labels.slot is not None
    assert labels.slot.tolist() == [[0, 1]]
    assert labels.move_source is None
    assert labels.attack_type is not None
    assert labels.attack_type.tolist() == [[-1, 0]]
    assert labels.action is not None
    assert labels.action.tolist() == [[0, 11]]
    assert labels.valid is not None
    assert labels.valid.dtype == torch.bool
    assert labels.valid.tolist() == [[True, False]]


def test_resolve_impala_teacher_auxiliary_coefficients_names_all_teacher_knobs() -> None:
    learner = ImpalaLearner(
        teacher_family_coef=0.1,
        teacher_slot_coef=0.2,
        teacher_hand_coef=0.3,
        teacher_move_source_coef=0.4,
        teacher_attack_type_coef=0.5,
        teacher_action_coef=0.6,
        teacher_same_family_action_coef=0.7,
        teacher_action_margin_coef=0.8,
        teacher_action_margin=0.9,
        teacher_same_family_action_margin_coef=1.1,
        teacher_same_family_action_margin=1.2,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=1.3,
        teacher_public_heuristic_temperature=1.4,
        teacher_public_nonpass_over_pass_coef=1.5,
        teacher_public_nonpass_over_pass_margin=1.6,
        teacher_public_heuristic_families=("main_play_character",),
    )

    coefficients = resolve_impala_teacher_auxiliary_coefficients(learner)

    assert coefficients.family == pytest.approx(0.1)
    assert coefficients.slot == pytest.approx(0.2)
    assert coefficients.hand == pytest.approx(0.3)
    assert coefficients.move_source == pytest.approx(0.4)
    assert coefficients.attack_type == pytest.approx(0.5)
    assert coefficients.action == pytest.approx(0.6)
    assert coefficients.same_family_action == pytest.approx(0.7)
    assert coefficients.action_margin == pytest.approx(0.8)
    assert coefficients.action_margin_value == pytest.approx(0.9)
    assert coefficients.same_family_action_margin == pytest.approx(1.1)
    assert coefficients.same_family_action_margin_value == pytest.approx(1.2)
    assert coefficients.exact_action_families == ("attack",)
    assert coefficients.public_heuristic == pytest.approx(1.3)
    assert coefficients.public_heuristic_temperature == pytest.approx(1.4)
    assert coefficients.public_nonpass_over_pass == pytest.approx(1.5)
    assert coefficients.public_nonpass_over_pass_margin == pytest.approx(1.6)
    assert coefficients.public_heuristic_families == ("main_play_character",)


def test_resolve_impala_teacher_auxiliary_factorized_inputs_preserves_required_and_optional_fields() -> None:
    required = {
        "family_log_probs": torch.zeros((1, 1, 2)),
        "play_slot_log_probs": torch.ones((1, 1, 3)),
        "move_slot_log_probs": torch.full((1, 1, 4), 2.0),
        "attack_slot_log_probs": torch.full((1, 1, 5), 3.0),
        "attack_type_log_probs": torch.full((1, 1, 6), 4.0),
    }
    result = resolve_impala_teacher_auxiliary_factorized_inputs(SimpleNamespace(**required))

    assert result.family_log_probs is required["family_log_probs"]
    assert result.play_slot_log_probs is required["play_slot_log_probs"]
    assert result.move_source_log_probs is None
    assert result.move_slot_log_probs is required["move_slot_log_probs"]
    assert result.attack_slot_log_probs is required["attack_slot_log_probs"]
    assert result.attack_type_log_probs is required["attack_type_log_probs"]
    assert result.top_action_ids is None
    assert result.same_family_action_logp is None
    assert result.same_family_top_action_ids is None


def test_resolve_impala_teacher_auxiliary_packed_inputs_preserves_tuple_contract() -> None:
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()

    packed = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
    )
    dense = resolve_impala_teacher_auxiliary_packed_inputs(
        packed_legal=None,
        packed_view=packed_view,
    )

    assert packed.ids is ids
    assert packed.offsets is offsets
    assert packed.meta is meta
    assert packed.view is packed_view
    assert dense.ids is None
    assert dense.offsets is None
    assert dense.meta is None
    assert dense.view is packed_view


def test_resolve_impala_teacher_auxiliary_inputs_preserves_aggregate_contract() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.25,
        teacher_public_heuristic_coef=0.75,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        same_family_action_logp=torch.tensor([[-0.5]]),
    )
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }

    result = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch=batch,
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )

    assert result.labels.family is not None
    assert result.labels.family.tolist() == [[family_index["main_play_character"]]]
    assert result.labels.move_source is None
    assert result.coefficients.family == pytest.approx(0.25)
    assert result.coefficients.public_heuristic == pytest.approx(0.75)
    assert result.coefficients.public_heuristic_families == ("main_play_character",)
    assert result.packed.ids is ids
    assert result.packed.offsets is offsets
    assert result.packed.meta is meta
    assert result.packed.view is packed_view
    assert result.factorized.family_log_probs is factorized_result.family_log_probs
    assert result.factorized.play_slot_log_probs is factorized_result.play_slot_log_probs
    assert result.factorized.same_family_action_logp is factorized_result.same_family_action_logp
    assert result.factorized.same_family_top_action_ids is None
