from __future__ import annotations

from .test_impala_learner import (
    Any,
    FactorizedStructuredTeacherModel,
    ImpalaLearner,
    LegalActionBatch,
    PairedOutcomeCandidateLogps,
    SimpleNamespace,
    TinyPolicyValueModel,
    TinyStructuredTeacherModel,
    VTraceTargets,
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _packed_structured_legal_view,
    _teacher_aux_catalog,
    _teacher_aux_hand_catalog,
    apply_impala_policy_anchor_stage,
    build_paired_outcome_preference_context,
    build_paired_outcome_preference_metrics,
    build_paired_swing_auxiliary_metrics,
    cast,
    compute_factorized_teacher_action_supervision,
    compute_factorized_teacher_group_supervision,
    compute_factorized_teacher_hand_supervision,
    compute_packed_structured_teacher_auxiliary_metrics,
    compute_packed_teacher_action_supervision,
    compute_packed_teacher_group_supervision,
    compute_packed_teacher_margin_supervision,
    compute_packed_teacher_public_supervision,
    compute_paired_outcome_candidate_logps,
    compute_paired_swing_candidate_view,
    compute_structured_teacher_auxiliary_metrics,
    empty_structured_teacher_metrics,
    impala_teacher_auxiliary_call,
    np,
    pytest,
    resolve_impala_teacher_auxiliary_inputs,
    resolve_paired_auxiliary_batch_inputs,
    resolve_structured_teacher_branch,
    resolve_structured_teacher_dispatch,
    resolve_structured_teacher_required_labels,
    resolve_structured_teacher_zero_context,
    structured_catalog_metadata,
    torch,
)


def test_apply_impala_policy_anchor_stage_preserves_inputs_loss_and_metrics() -> None:
    anchor_loss = torch.tensor(0.75, dtype=torch.float32)
    calls: list[dict[str, Any]] = []
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_model = object()
    reset_before_step = torch.tensor([[False], [True]], dtype=torch.bool)
    inputs = SimpleNamespace(
        obs=obs,
        loss_mask=loss_mask,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_model=forward_model,
        reset_before_step=reset_before_step,
    )
    batch: dict[str, bool] = {"policy_anchor_batch": True}

    def fake_policy_anchor_loss_and_metrics(
        source_batch: Any,
        *,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
        factorized_result: Any,
        forward_model: Any,
        reset_before_step: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        calls.append(
            {
                "batch": source_batch,
                "obs": obs,
                "loss_mask": loss_mask,
                "packed_legal": packed_legal,
                "factorized_result": factorized_result,
                "forward_model": forward_model,
                "reset_before_step": reset_before_step,
            }
        )
        return anchor_loss, {"policy_anchor_weighted_loss": float(anchor_loss)}

    learner = SimpleNamespace(_policy_anchor_loss_and_metrics=fake_policy_anchor_loss_and_metrics)
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss + anchor_loss)
    assert result.policy_anchor_loss is anchor_loss
    assert result.policy_anchor_metrics["policy_anchor_weighted_loss"] == pytest.approx(0.75)
    assert calls == [
        {
            "batch": batch,
            "obs": obs,
            "loss_mask": loss_mask,
            "packed_legal": packed_legal,
            "factorized_result": factorized_result,
            "forward_model": forward_model,
            "reset_before_step": reset_before_step,
        }
    ]


def test_apply_impala_policy_anchor_stage_preserves_total_loss_when_anchor_disabled() -> None:
    learner = SimpleNamespace(
        _policy_anchor_loss_and_metrics=lambda *args, **kwargs: (
            None,
            {"policy_anchor_disabled": 1.0},
        )
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        packed_legal=None,
        factorized_result=None,
        forward_model=object(),
        reset_before_step=None,
    )
    base_loss = torch.tensor(2.0, dtype=torch.float32)

    result = apply_impala_policy_anchor_stage(
        learner=learner,
        batch={},
        inputs=cast(Any, inputs),
        total_loss=base_loss,
    )

    torch.testing.assert_close(result.total_loss, base_loss)
    assert result.policy_anchor_loss is None
    assert result.policy_anchor_metrics == {"policy_anchor_disabled": 1.0}


def test_compute_structured_teacher_auxiliary_metrics_supervises_slot_groups_not_hand_indices() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Row 0: two different hand indices map to the same play slot. Slot supervision should
    # treat their combined probability mass as correct.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1: attack family with the correct attack type.
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        teacher_slot=torch.tensor([[0], [0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [11]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.1,
        attack_type_coef=0.05,
        action_coef=0.15,
        same_family_action_coef=0.2,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_loss"] < 0.05
    assert metrics["teacher_action_loss"] < 0.35
    assert metrics["teacher_same_family_action_loss"] < 0.35


def test_compute_structured_teacher_auxiliary_metrics_groups_main_move_targets_by_destination_slot() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_actions_by_target: dict[int, list[int]] = {}
    for action_id in range(action_catalog.action_space_size):
        decoded = action_catalog.decode(action_id)
        if decoded.family != "main_move" or decoded.to_slot is None:
            continue
        move_actions_by_target.setdefault(int(decoded.to_slot), []).append(int(action_id))
    target_slot, target_actions = next(
        (slot, action_ids) for slot, action_ids in move_actions_by_target.items() if len(action_ids) >= 2
    )
    preferred_move, alternate_move = target_actions[:2]

    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [preferred_move, alternate_move, action_catalog.pass_action_id]] = True
    logits[0, 0, preferred_move] = 1.0
    logits[0, 0, alternate_move] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -4.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        "teacher_slot": torch.tensor([[target_slot]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[preferred_move]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 1.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 1.0,
    }

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)
    assert dense_metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert dense_metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert dense_metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)

    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
    }

    logits[0, 0, 0] = 4.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -5.0
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    logits[0, 0, 0] = 0.5
    logits[0, 0, 5] = 4.0
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_from_impala_inputs_maps_all_fields(monkeypatch) -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.11,
        teacher_slot_coef=0.12,
        teacher_hand_coef=0.13,
        teacher_move_source_coef=0.14,
        teacher_attack_type_coef=0.15,
        teacher_action_coef=0.16,
        teacher_same_family_action_coef=0.17,
        teacher_action_margin_coef=0.18,
        teacher_action_margin=0.19,
        teacher_same_family_action_margin_coef=0.20,
        teacher_same_family_action_margin=0.21,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=0.22,
        teacher_public_heuristic_temperature=0.23,
        teacher_public_nonpass_over_pass_coef=0.24,
        teacher_public_nonpass_over_pass_margin=0.25,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_source_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 1.5),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        top_action_ids=torch.tensor([[0]], dtype=torch.long),
        same_family_action_logp=torch.tensor([[-0.5]]),
        same_family_top_action_ids=torch.tensor([[5]], dtype=torch.long),
        same_family_arg0_logp=torch.tensor([[-0.6]]),
        same_family_top_arg0=torch.tensor([[1]], dtype=torch.long),
    )
    inputs = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_move_source": np.asarray([[1]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    public_target = torch.arange(2, dtype=torch.float32)
    sentinel_loss = torch.tensor(9.0)
    sentinel_metrics = {"teacher_aux_loss": 9.0}
    sentinel_context = {"teacher_family_log_probs": torch.tensor([1.0])}
    captured: dict[str, Any] = {}

    def fake_compute_structured_teacher_auxiliary_metrics(
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        captured.update(kwargs)
        return sentinel_loss, sentinel_metrics, sentinel_context

    monkeypatch.setattr(
        impala_teacher_auxiliary_call,
        "compute_structured_teacher_auxiliary_metrics",
        fake_compute_structured_teacher_auxiliary_metrics,
    )

    loss, metrics, context = impala_teacher_auxiliary_call.compute_structured_teacher_auxiliary_from_impala_inputs(
        inputs=inputs,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        public_heuristic_target_logits=public_target,
    )

    assert loss is sentinel_loss
    assert metrics is sentinel_metrics
    assert context is sentinel_context
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["teacher_family"] is inputs.labels.family
    assert captured["teacher_slot"] is inputs.labels.slot
    assert captured["teacher_move_source"] is inputs.labels.move_source
    assert captured["teacher_attack_type"] is inputs.labels.attack_type
    assert captured["teacher_action"] is inputs.labels.action
    assert captured["teacher_valid"] is inputs.labels.valid
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] is action_catalog
    assert captured["family_coef"] == pytest.approx(0.11)
    assert captured["slot_coef"] == pytest.approx(0.12)
    assert captured["hand_coef"] == pytest.approx(0.13)
    assert captured["move_source_coef"] == pytest.approx(0.14)
    assert captured["attack_type_coef"] == pytest.approx(0.15)
    assert captured["action_coef"] == pytest.approx(0.16)
    assert captured["same_family_action_coef"] == pytest.approx(0.17)
    assert captured["action_margin_coef"] == pytest.approx(0.18)
    assert captured["action_margin"] == pytest.approx(0.19)
    assert captured["same_family_action_margin_coef"] == pytest.approx(0.20)
    assert captured["same_family_action_margin"] == pytest.approx(0.21)
    assert captured["exact_action_families"] == ("attack",)
    assert captured["public_heuristic_coef"] == pytest.approx(0.22)
    assert captured["public_heuristic_temperature"] == pytest.approx(0.23)
    assert captured["public_nonpass_over_pass_coef"] == pytest.approx(0.24)
    assert captured["public_nonpass_over_pass_margin"] == pytest.approx(0.25)
    assert captured["public_heuristic_families"] == ("main_play_character",)
    assert captured["public_heuristic_target_logits"] is public_target
    assert captured["packed_ids"] is ids
    assert captured["packed_offsets"] is offsets
    assert captured["packed_meta"] is meta
    assert captured["packed_view"] is packed_view
    assert captured["factorized_family_log_probs"] is factorized_result.family_log_probs
    assert captured["factorized_play_slot_log_probs"] is factorized_result.play_slot_log_probs
    assert captured["factorized_move_source_log_probs"] is factorized_result.move_source_log_probs
    assert captured["factorized_move_slot_log_probs"] is factorized_result.move_slot_log_probs
    assert captured["factorized_attack_slot_log_probs"] is factorized_result.attack_slot_log_probs
    assert captured["factorized_attack_type_log_probs"] is factorized_result.attack_type_log_probs
    assert captured["factorized_top_action_ids"] is factorized_result.top_action_ids
    assert captured["factorized_same_family_action_logp"] is factorized_result.same_family_action_logp
    assert captured["factorized_same_family_top_action_ids"] is factorized_result.same_family_top_action_ids
    assert captured["factorized_same_family_arg0_logp"] is factorized_result.same_family_arg0_logp
    assert captured["factorized_same_family_top_arg0"] is factorized_result.same_family_top_arg0


def test_resolve_structured_teacher_zero_context_uses_packed_view_before_loss_mask() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([1.0, 2.0], dtype=torch.float64),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    packed_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=packed_view,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )
    mask_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )

    assert packed_zero.value_dtype == packed_view.logits.dtype
    assert packed_zero.zero.dtype == packed_view.logits.dtype
    assert packed_zero.empty_metrics["teacher_aux_loss"] == pytest.approx(0.0)
    assert mask_zero.value_dtype == torch.float64


def test_resolve_structured_teacher_required_labels_names_missing_label_gate() -> None:
    family = torch.tensor([[0]], dtype=torch.long)
    slot = torch.tensor([[1]], dtype=torch.long)
    attack_type = torch.tensor([[-1]], dtype=torch.long)
    valid = torch.tensor([[True]], dtype=torch.bool)

    labels = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=slot,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )
    missing = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=None,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )

    assert labels is not None
    assert labels.family is family
    assert labels.slot is slot
    assert labels.attack_type is attack_type
    assert labels.valid is valid
    assert missing is None


def test_resolve_structured_teacher_branch_prioritizes_factorized_then_packed_then_dense() -> None:
    factorized = resolve_structured_teacher_branch(
        factorized_family_log_probs=torch.zeros((1, 1, 2)),
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    packed = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    dense = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    inactive = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=None,
    )

    assert factorized.use_factorized is True
    assert factorized.use_packed is False
    assert factorized.use_dense is False
    assert packed.use_factorized is False
    assert packed.use_packed is True
    assert packed.use_dense is False
    assert dense.use_factorized is False
    assert dense.use_packed is False
    assert dense.use_dense is True
    assert inactive.use_factorized is False
    assert inactive.use_packed is False
    assert inactive.use_dense is False


def test_resolve_structured_teacher_dispatch_preserves_label_gate_before_packed_view_build() -> None:
    action_catalog = _teacher_aux_catalog()
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float64)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    invalid_packed_meta = torch.zeros((2, 3), dtype=torch.long)
    labels = {
        "teacher_family": torch.tensor([[0]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
    }

    missing_label_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=invalid_packed_meta,
        packed_view=None,
        factorized_family_log_probs=None,
        teacher_family=labels["teacher_family"],
        teacher_slot=None,
        teacher_attack_type=labels["teacher_attack_type"],
        teacher_valid=labels["teacher_valid"],
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
    )

    assert missing_label_dispatch.labels is None
    assert missing_label_dispatch.packed_view is None
    assert missing_label_dispatch.branch.use_factorized is False
    assert missing_label_dispatch.branch.use_packed is False
    assert missing_label_dispatch.branch.use_dense is False
    assert missing_label_dispatch.zero_context.value_dtype == torch.float64

    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    factorized_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=None,
        factorized_family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        **labels,
    )

    assert factorized_dispatch.labels is not None
    assert factorized_dispatch.packed_view is not None
    assert factorized_dispatch.branch.use_factorized is True
    assert factorized_dispatch.branch.use_packed is False
    assert factorized_dispatch.branch.use_dense is False
    assert factorized_dispatch.zero_context.value_dtype == torch.float64


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_public_heuristic_soft_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    teacher_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    misaligned_view = _packed_structured_legal_view(
        logits=torch.tensor([4.0, 0.5, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    misaligned_loss, misaligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=misaligned_view,
        **cast(Any, teacher_kwargs),
    )

    aligned_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    aligned_loss, aligned_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        packed_view=aligned_view,
        **cast(Any, teacher_kwargs),
    )

    assert float(misaligned_loss.detach()) > float(aligned_loss.detach())
    assert misaligned_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert aligned_metrics["teacher_public_heuristic_loss"] < misaligned_metrics["teacher_public_heuristic_loss"]
    assert (
        aligned_metrics["teacher_public_heuristic_top1_mass"] > misaligned_metrics["teacher_public_heuristic_top1_mass"]
    )


def test_compute_structured_teacher_auxiliary_metrics_gates_public_heuristic_by_family() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    public_target_logits = torch.tensor([0.0, 3.0, -6.0], dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 4.0
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.5, 4.0, -5.0], dtype=torch.float32),
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    common_kwargs = {
        "logits": None,
        "legal_mask": None,
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[0]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.0,
        "slot_coef": 0.0,
        "attack_type_coef": 0.0,
        "action_coef": 0.0,
        "same_family_action_coef": 0.0,
        "public_heuristic_coef": 1.0,
        "public_heuristic_temperature": 1.0,
        "public_heuristic_target_logits": public_target_logits,
        "packed_ids": torch.as_tensor(packed_ids, dtype=torch.long),
        "packed_offsets": torch.as_tensor(packed_offsets, dtype=torch.long),
        "packed_meta": torch.as_tensor(packed_meta, dtype=torch.long),
        "packed_view": packed_view,
        "factorized_family_log_probs": torch.log_softmax(family_logits, dim=-1),
    }

    allowed_loss, allowed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("main_play_character",),
        **cast(Any, common_kwargs),
    )
    gated_loss, gated_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        public_heuristic_families=("attack",),
        **cast(Any, common_kwargs),
    )

    assert float(allowed_loss.detach()) > 0.0
    assert allowed_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert allowed_metrics["teacher_public_heuristic_loss"] > 0.0
    assert float(gated_loss.detach()) == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(0.0)
    assert gated_metrics["teacher_public_heuristic_loss"] == pytest.approx(0.0)


def test_compute_structured_teacher_auxiliary_metrics_matches_packed_meta_path() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0
    legal_mask[1, 0, [10, 11, 12, 19]] = True
    logits[1, 0, 10] = 0.5
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = 0.0
    logits[1, 0, 19] = -3.0
    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_loss, dense_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        **cast(Any, teacher_kwargs),
    )
    packed_loss, packed_metrics, _ = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(dense_loss, packed_loss)
    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_packed_structured_teacher_auxiliary_metrics_matches_dispatcher() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_meta_tensor = torch.as_tensor(packed_meta, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
    )
    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[5]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.0,
        "action_coef": 0.3,
        "same_family_action_coef": 0.4,
    }

    dispatch_loss, dispatch_metrics, dispatch_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
        packed_view=packed_view,
        **cast(Any, teacher_kwargs),
    )
    direct_loss, direct_metrics, direct_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_move_source=None,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(direct_loss, dispatch_loss)
    assert direct_metrics == pytest.approx(dispatch_metrics)
    assert direct_context.keys() == dispatch_context.keys()
    for key in direct_context:
        torch.testing.assert_close(direct_context[key], dispatch_context[key])


def test_compute_packed_teacher_action_supervision_matches_packed_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[5]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    direct = compute_packed_teacher_action_supervision(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_catalog=action_catalog,
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=1.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )

    torch.testing.assert_close(packed_loss, direct.action_loss + direct.same_family_action_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_action_supervision_matches_factorized_branch_action_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -3.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_log_probs = torch.log_softmax(family_logits, dim=-1)
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[0], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    same_family_logp = torch.tensor([[-0.1], [-0.4]], dtype=torch.float32)
    same_family_top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    top_action_ids = torch.tensor([[0], [move_action]], dtype=torch.long)
    zero = family_log_probs.sum() * 0.0

    direct = compute_factorized_teacher_action_supervision(
        family_log_probs=family_log_probs.reshape(-1, family_log_probs.shape[-1]),
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        action_coef=1.0,
        same_family_action_coef=1.0,
        zero=zero,
        value_dtype=family_log_probs.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.3,
        same_family_action_coef=0.7,
        factorized_family_log_probs=family_log_probs,
        factorized_top_action_ids=top_action_ids,
        factorized_same_family_action_logp=same_family_logp,
        factorized_same_family_top_action_ids=same_family_top_action_ids,
    )
    expected_action_loss = direct.action_loss * 0.3 + direct.same_family_action_loss * 0.7

    torch.testing.assert_close(factorized_loss, expected_action_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)


def test_compute_packed_teacher_group_supervision_matches_packed_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((3, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((3, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[1, 0, [10, 11, 12, action_catalog.pass_action_id]] = True
    logits[1, 0, 10] = -2.0
    logits[1, 0, 11] = 4.0
    logits[1, 0, 12] = -1.0
    logits[1, 0, action_catalog.pass_action_id] = -3.0
    legal_mask[2, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[2, 0, move_action] = 3.5
    logits[2, 0, competing_move_action] = -0.5
    logits[2, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["attack"]], [family_index["main_move"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [0], [int(move_decoded.to_slot or 0)]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [attack_type_index["direct"]], [-1]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [11], [move_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    teacher_move_source = torch.tensor([[-1], [-1], [int(move_decoded.from_slot or 0)]], dtype=torch.long)
    loss_mask = torch.ones((3, 1), dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)

    direct = compute_packed_teacher_group_supervision(
        packed_view=packed_view,
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=teacher_move_source.reshape(-1),
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        action_catalog=action_catalog,
        family_names=metadata.family_names,
        family_index={name: index for index, name in enumerate(metadata.family_names)},
        attack_type_names=metadata.attack_type_names,
        move_source_targets_by_action=None,
        move_source_coef=1.0,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=teacher_move_source,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.5,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(packed_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_factorized_teacher_group_supervision_matches_factorized_branch_group_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    row_count = 3
    family_logits = torch.full((row_count, 1, len(action_catalog.families)), -4.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    family_logits[2, 0, family_index["attack"]] = 3.0
    play_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    play_slot_logits[0, 0, 0] = 3.0
    move_slot_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_slot_logits[1, 0, int(move_decoded.to_slot or 0)] = 3.0
    move_source_logits = torch.full((row_count, 1, int(action_catalog.max_stage)), -4.0)
    move_source_logits[1, 0, int(move_decoded.from_slot or 0)] = 3.0
    attack_slot_logits = torch.zeros((row_count, 1, int(action_catalog.attack_slot_count)), dtype=torch.float32)
    attack_type_logits = torch.full((row_count, 1, len(action_catalog.attack_type_names)), -4.0)
    attack_type_logits[2, 0, attack_type_index["direct"]] = 3.0
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["main_move"]], [family_index["attack"]]],
        dtype=torch.long,
    )
    teacher_slot = torch.tensor([[0], [int(move_decoded.to_slot or 0)], [0]], dtype=torch.long)
    teacher_attack_type = torch.tensor([[-1], [-1], [attack_type_index["direct"]]], dtype=torch.long)
    teacher_action = torch.tensor([[0], [move_action], [10]], dtype=torch.long)
    teacher_valid = torch.ones((row_count, 1), dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5], [0.25]], dtype=torch.float32)
    metadata = structured_catalog_metadata(action_catalog)
    zero = family_logits.sum() * 0.0

    direct = compute_factorized_teacher_group_supervision(
        family_log_probs=torch.log_softmax(family_logits, dim=-1).reshape(row_count, -1),
        play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1).reshape(row_count, -1),
        move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1).reshape(row_count, -1),
        move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1).reshape(row_count, -1),
        attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1).reshape(row_count, -1),
        attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1).reshape(row_count, -1),
        flat_loss_mask=loss_mask.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_slot=teacher_slot.reshape(-1),
        flat_teacher_move_source=None,
        flat_teacher_attack_type=teacher_attack_type.reshape(-1),
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        attack_type_names=tuple(action_catalog.attack_type_names),
        move_source_targets_by_action=torch.as_tensor(metadata.move_from_slots, dtype=torch.long),
        play_family_id=family_index["main_play_character"],
        move_family_id=family_index["main_move"],
        attack_family_id=family_index["attack"],
        move_source_coef=1.0,
        zero=zero,
        value_dtype=family_logits.dtype,
    )
    factorized_loss, factorized_metrics, factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=teacher_slot,
        teacher_attack_type=teacher_attack_type,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.2,
        slot_coef=0.3,
        attack_type_coef=0.4,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.5,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_play_slot_log_probs=torch.log_softmax(play_slot_logits, dim=-1),
        factorized_move_source_log_probs=torch.log_softmax(move_source_logits, dim=-1),
        factorized_move_slot_log_probs=torch.log_softmax(move_slot_logits, dim=-1),
        factorized_attack_slot_log_probs=torch.log_softmax(attack_slot_logits, dim=-1),
        factorized_attack_type_log_probs=torch.log_softmax(attack_type_logits, dim=-1),
    )
    expected_group_loss = (
        direct.family_loss * 0.2
        + direct.slot_loss * 0.3
        + direct.attack_type_loss * 0.4
        + direct.move_source_loss * 0.5
    )

    torch.testing.assert_close(factorized_loss, expected_group_loss)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(factorized_context[key], value)


def test_compute_packed_teacher_public_supervision_matches_packed_branch_public_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, -0.5, 3.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    target_logits = torch.tensor([4.0, 5.0, -5.0], dtype=torch.float32)

    direct = compute_packed_teacher_public_supervision(
        packed_view=packed_view,
        public_heuristic_target_logits=target_logits,
        public_heuristic_family_ids=(family_index["main_play_character"],),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        pass_action_id=action_catalog.pass_action_id,
        public_heuristic_coef=1.0,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=1.0,
        public_nonpass_over_pass_margin=0.5,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=target_logits,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_public_loss = direct.public_heuristic_loss * 0.7 + direct.public_nonpass_over_pass_loss * 0.3

    torch.testing.assert_close(packed_loss, expected_public_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_compute_packed_teacher_margin_supervision_matches_packed_branch_margin_terms() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[5]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    direct = compute_packed_teacher_margin_supervision(
        packed_view=packed_view,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        action_margin_coef=1.0,
        action_margin=0.5,
        same_family_action_margin_coef=1.0,
        same_family_action_margin=0.5,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
    )
    packed_loss, packed_metrics, packed_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        teacher_move_source=None,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.25,
        action_margin=0.5,
        same_family_action_margin_coef=0.75,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
    )
    expected_margin_loss = direct.action_margin_loss * 0.25 + direct.same_family_action_margin_loss * 0.75

    torch.testing.assert_close(packed_loss, expected_margin_loss)
    for key, value in direct.metrics.items():
        assert packed_metrics[key] == pytest.approx(value)
    for key, value in direct.context.items():
        torch.testing.assert_close(packed_context[key], value)


def test_factorized_structured_teacher_reuses_packed_public_and_margin_helpers() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[0]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -3.0, dtype=torch.float32)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    public_target_logits = torch.tensor([4.0, 5.0, -5.0], dtype=torch.float32)
    zero = packed_view.logits.sum() * 0.0
    margin_direct = compute_packed_teacher_margin_supervision(
        packed_view=packed_view,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        action_margin_coef=1.0,
        action_margin=0.5,
        same_family_action_margin_coef=1.0,
        same_family_action_margin=0.5,
        zero=zero,
        value_dtype=packed_view.logits.dtype,
    )
    public_direct = compute_packed_teacher_public_supervision(
        packed_view=packed_view,
        public_heuristic_target_logits=public_target_logits,
        public_heuristic_family_ids=(family_index["main_play_character"],),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        pass_action_id=action_catalog.pass_action_id,
        public_heuristic_coef=1.0,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=1.0,
        public_nonpass_over_pass_margin=0.5,
        zero=zero,
        value_dtype=packed_view.logits.dtype,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=0.2,
        action_margin=0.5,
        same_family_action_margin_coef=0.4,
        same_family_action_margin=0.5,
        public_heuristic_coef=0.7,
        public_heuristic_temperature=1.0,
        public_nonpass_over_pass_coef=0.3,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=("main_play_character",),
        public_heuristic_target_logits=public_target_logits,
        packed_view=packed_view,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )
    expected_loss = (
        margin_direct.action_margin_loss * 0.2
        + margin_direct.same_family_action_margin_loss * 0.4
        + public_direct.public_heuristic_loss * 0.7
        + public_direct.public_nonpass_over_pass_loss * 0.3
    )

    torch.testing.assert_close(aux_loss, expected_loss)
    expected_metrics = {**margin_direct.metrics, **public_direct.metrics}
    for key, value in expected_metrics.items():
        assert metrics[key] == pytest.approx(value)
    expected_context = {**margin_direct.context, **public_direct.context}
    for key, value in expected_context.items():
        torch.testing.assert_close(context[key], value)


def test_compute_structured_teacher_auxiliary_metrics_infers_packed_move_source_from_action() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    competing_move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 1)
    )
    move_decoded = action_catalog.decode(move_action)
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [move_action, competing_move_action, action_catalog.pass_action_id]] = True
    logits[0, 0, move_action] = 4.0
    logits[0, 0, competing_move_action] = -1.0
    logits[0, 0, action_catalog.pass_action_id] = -3.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_skips_unsupported_packed_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((2, 1, action_catalog.action_space_size), dtype=torch.bool)

    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 2.5
    logits[0, 0, 19] = -4.0

    # Row 1 carries attack teacher labels but only exposes pass legally, which previously
    # produced NaNs in the packed grouped-log-prob path.
    legal_mask[1, 0, [19]] = True
    logits[1, 0, 19] = 1.0

    teacher_kwargs = {
        "teacher_family": torch.tensor(
            [[family_index["main_play_character"]], [family_index["attack"]]], dtype=torch.long
        ),
        "teacher_slot": torch.tensor([[0], [0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1], [attack_type_index["direct"]]], dtype=torch.long),
        "teacher_action": torch.tensor([[0], [11]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True], [True]], dtype=torch.bool),
        "loss_mask": torch.ones((2, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.05,
        "action_coef": 0.15,
        "same_family_action_coef": 0.2,
    }
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    packed_loss, packed_metrics, packed_context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
        **cast(Any, teacher_kwargs),
    )

    assert torch.isfinite(packed_loss)
    assert np.isfinite(packed_metrics["teacher_aux_loss"])
    assert np.isfinite(packed_metrics["teacher_family_loss"])
    assert np.isfinite(packed_metrics["teacher_slot_loss"])
    assert np.isfinite(packed_metrics["teacher_attack_type_loss"])
    assert np.isfinite(packed_metrics["teacher_action_loss"])
    assert np.isfinite(packed_metrics["teacher_same_family_action_loss"])
    assert packed_metrics["teacher_action_supported_fraction"] == pytest.approx(0.5)
    assert packed_metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(0.5)
    assert "teacher_attack_type_log_probs" not in packed_context
    assert "teacher_family_log_probs" in packed_context
    assert "teacher_action_log_probs" in packed_context
    assert "teacher_same_family_action_log_probs" in packed_context
    assert not torch.isnan(packed_context["teacher_family_log_probs"]).any()


def test_compute_structured_teacher_auxiliary_metrics_reports_within_family_tactical_miss() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)

    # Two play-character actions share the same play slot. The model picks the wrong hand index,
    # so family and slot stay correct while the exact within-family choice is wrong.
    legal_mask[0, 0, [0, 5, 19]] = True
    logits[0, 0, 0] = 1.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, 19] = -4.0

    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
    )

    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_same_family_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_move"
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["main_move"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_action_logp=torch.tensor([[-0.1], [-0.2]], dtype=torch.float32),
        factorized_same_family_top_action_ids=torch.tensor([[0], [move_action]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_move_accuracy"] == pytest.approx(1.0)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_exact_action_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_top_action_ids=torch.tensor([[0]], dtype=torch.long),
        factorized_same_family_action_logp=torch.tensor([[-0.1]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_action_loss"] > 0.0


def test_compute_factorized_teacher_hand_supervision_matches_factorized_branch_hand_terms() -> None:
    action_catalog = _teacher_aux_hand_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    play_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_play_character"
        and action_catalog.decode(action_id).hand_index is not None
    )
    clock_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "clock_from_hand"
        and action_catalog.decode(action_id).hand_index is not None
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["clock_from_hand"]] = 3.0
    teacher_family = torch.tensor(
        [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
        dtype=torch.long,
    )
    teacher_action = torch.tensor([[play_action], [clock_action]], dtype=torch.long)
    teacher_valid = torch.tensor([[True], [True]], dtype=torch.bool)
    loss_mask = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    arg0_logp = torch.tensor([[-0.05], [-0.20]], dtype=torch.float32)
    top_arg0 = torch.tensor([[play_hand], [clock_hand]], dtype=torch.long)
    metadata = structured_catalog_metadata(action_catalog)
    zero = family_logits.sum() * 0.0

    direct = compute_factorized_teacher_hand_supervision(
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
        flat_teacher_action=teacher_action.reshape(-1),
        flat_teacher_family=teacher_family.reshape(-1),
        flat_teacher_valid=teacher_valid.reshape(-1),
        flat_loss_mask=loss_mask.reshape(-1),
        exact_action_family_rows=None,
        hand_targets_by_action=torch.as_tensor(metadata.hand_indices, dtype=torch.long),
        hand_family_ids=(family_index["main_play_character"], family_index["clock_from_hand"]),
        play_family_id=family_index["main_play_character"],
        clock_from_hand_family_id=family_index["clock_from_hand"],
        hand_coef=1.0,
        zero=zero,
        value_dtype=family_logits.dtype,
    )
    factorized_loss, factorized_metrics, _factorized_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=teacher_family,
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=0.4,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_arg0_logp=arg0_logp,
        factorized_same_family_top_arg0=top_arg0,
    )

    torch.testing.assert_close(factorized_loss, direct.hand_loss * 0.4)
    for key, value in direct.metrics.items():
        assert factorized_metrics[key] == pytest.approx(value)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_hand_targets() -> None:
    action_catalog = _teacher_aux_hand_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    play_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "main_play_character"
        and action_catalog.decode(action_id).hand_index is not None
    )
    clock_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if action_catalog.decode(action_id).family == "clock_from_hand"
        and action_catalog.decode(action_id).hand_index is not None
    )
    play_hand = int(action_catalog.decode(play_action).hand_index or 0)
    clock_hand = int(action_catalog.decode(clock_action).hand_index or 0)
    family_logits = torch.full((2, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_play_character"]] = 3.0
    family_logits[1, 0, family_index["clock_from_hand"]] = 3.0

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [[family_index["main_play_character"]], [family_index["clock_from_hand"]]],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [-1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[play_action], [clock_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True]], dtype=torch.bool),
        loss_mask=torch.ones((2, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        hand_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_same_family_arg0_logp=torch.tensor([[-0.05], [-0.10]], dtype=torch.float32),
        factorized_same_family_top_arg0=torch.tensor([[play_hand], [clock_hand]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_clock_from_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_hand_loss"] == pytest.approx(0.075)


def test_compute_structured_teacher_auxiliary_metrics_supports_factorized_move_source_targets() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[move_action]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_supports_explicit_move_source_labels() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    family_logits = torch.full((1, 1, len(action_catalog.families)), -2.0)
    family_logits[0, 0, family_index["main_move"]] = 3.0
    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([[family_index["main_move"]]], dtype=torch.long),
        teacher_slot=torch.tensor([[int(move_decoded.to_slot or 0)]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1]], dtype=torch.long),
        teacher_action=torch.tensor([[-1]], dtype=torch.long),
        teacher_valid=torch.tensor([[True]], dtype=torch.bool),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=1.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
        factorized_move_source_log_probs=torch.tensor([[[-0.01, -5.0, -5.0, -5.0, -5.0]]], dtype=torch.float32),
        teacher_move_source=torch.tensor([[int(move_decoded.from_slot or 0)]], dtype=torch.long),
    )

    assert float(aux_loss.detach()) > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_loss"] > 0.0


def test_compute_structured_teacher_auxiliary_metrics_reports_family_coverage_on_active_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    family_logits = torch.zeros((4, 1, len(action_catalog.families)), dtype=torch.float32)
    _aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor(
            [
                [family_index["main_play_character"]],
                [family_index["main_move"]],
                [family_index["attack"]],
                [family_index["main_move"]],
            ],
            dtype=torch.long,
        ),
        teacher_slot=torch.tensor([[0], [1], [0], [2]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[-1], [-1], [0], [-1]], dtype=torch.long),
        teacher_action=torch.tensor([[0], [5], [11], [5]], dtype=torch.long),
        teacher_valid=torch.tensor([[True], [True], [True], [False]], dtype=torch.bool),
        loss_mask=torch.tensor([[1.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
        action_catalog=action_catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        move_source_coef=0.0,
        factorized_family_log_probs=torch.log_softmax(family_logits, dim=-1),
    )

    assert metrics["teacher_active_fraction"] == pytest.approx(0.75)
    assert metrics["teacher_main_play_character_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_main_move_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["teacher_attack_fraction"] == pytest.approx(0.0)


def test_impala_learner_auxiliary_update_uses_factorized_same_family_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_same_family_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_same_family_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_main_play_character_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_hand_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_hand_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_hand_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_hand_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_main_play_character_hand_accuracy"] == pytest.approx(1.0)


def test_impala_learner_factorized_policy_anchor_penalizes_post_anchor_drift() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(
        model=model,
        policy_anchor_coef=0.5,
        policy_anchor_temperature=1.0,
    )
    learner._ensure_policy_anchor_model()
    with torch.no_grad():
        model.bias.fill_(2.0)
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
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.zeros((2, 1), dtype=np.float32),
            pg_advantages=np.ones((2, 1), dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["policy_anchor_coef_active"] == pytest.approx(0.5)
    assert metrics["policy_anchor_loss"] > 0.0
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(metrics["policy_anchor_loss"] * 0.5)
    assert metrics["policy_anchor_candidate_count"] == pytest.approx(float(packed_ids.shape[0]))
    assert model.factorized_candidate_logp_calls == 1
    assert learner._policy_anchor_model is not None


def test_impala_learner_reset_policy_anchor_refreshes_current_weights() -> None:
    model = TinyPolicyValueModel()
    learner = ImpalaLearner(model=model, policy_anchor_coef=0.5)
    learner._ensure_policy_anchor_model()

    with torch.no_grad():
        model.policy.bias.fill_(3.0)
    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is not None
    anchor_bias = dict(learner._policy_anchor_model.state_dict())["policy.bias"]
    assert torch.equal(anchor_bias, model.policy.bias.detach())


def test_impala_learner_reset_policy_anchor_clears_disabled_anchor() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel())
    learner._ensure_policy_anchor_model()

    learner.reset_policy_anchor_to_current_model()

    assert learner._policy_anchor_model is None


def test_impala_learner_auxiliary_update_uses_factorized_teacher_action_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [0]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_move_source_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    move_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (action_catalog.decode(action_id).family == "main_move" and action_catalog.decode(action_id).from_slot == 0)
    )
    move_decoded = action_catalog.decode(move_action)
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_move_source_coef=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([move_action, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_move"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[int(move_decoded.to_slot or 0)]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[move_action]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_move_source_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_move_source_accuracy"] == pytest.approx(1.0)


def test_impala_learner_auxiliary_update_uses_factorized_public_heuristic_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.trunk_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_loss"] > 0.0
    assert metrics["teacher_public_heuristic_top1_mass"] < 0.1


def test_impala_learner_factorized_margin_aux_uses_factorized_candidate_log_probs_without_public_teacher() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_action_margin_coef=1.0,
        teacher_action_margin=0.5,
        teacher_same_family_action_margin_coef=1.0,
        teacher_same_family_action_margin=0.5,
        teacher_public_heuristic_coef=0.0,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert model.factorized_candidate_logp_calls == 1
    assert model.public_student_calls == 0
    assert model.public_target_calls == 0
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_action_margin_mean"] > 0.5
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.0)
    assert metrics["teacher_same_family_action_margin_mean"] > 0.5


def test_impala_learner_paired_swing_auxiliary_dense_path_preserves_weighted_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TinyStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = 0.0
        model.policy.bias[5] = 1.0
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "actions": np.asarray([[5]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=_packed_meta_from_ids(action_catalog, packed_ids),
            action_space=action_catalog.action_space_size,
        ),
    }

    loss, metrics, context = learner._paired_swing_loss_and_metrics(
        batch,
        margin=0.25,
        coef=0.5,
        positive_action_source="teacher_action",
        negative_action_source="actions",
    )

    assert loss.detach().item() == pytest.approx(0.625)
    assert metrics["paired_swing_weighted_loss"] == pytest.approx(0.625)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_rows"] == 1.0
    assert context["paired_swing_margins"].tolist() == pytest.approx([-1.0])


def test_compute_paired_swing_candidate_view_preserves_dense_path_outputs() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TinyStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = 0.0
        model.policy.bias[5] = 1.0
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5], dtype=np.uint32)
    packed_offsets = np.asarray([0, 2], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "actions": np.asarray([[5]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "legal_actions": LegalActionBatch.from_packed(
            packed_ids,
            packed_offsets,
            meta=_packed_meta_from_ids(action_catalog, packed_ids),
            action_space=action_catalog.action_space_size,
        ),
    }
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired-swing replay requires packed legal_ids/legal_offsets",
    )

    candidate_view = compute_paired_swing_candidate_view(
        learner,
        batch,
        obs=inputs.obs,
        expected_shape=inputs.expected_shape,
        packed_legal=inputs.packed_legal,
        loss_mask=inputs.loss_mask,
        margin_retention_coef=0.0,
        top_action_retention_coef=0.0,
    )

    assert candidate_view.reference_packed_logits is None
    assert candidate_view.logits is not None
    assert candidate_view.values is not None
    assert candidate_view.zero.item() == pytest.approx(0.0)
    assert candidate_view.packed_view.logits.tolist() == pytest.approx([0.0, 1.0])
    assert candidate_view.logits.shape == torch.Size([1, 1, action_catalog.action_space_size])
    assert candidate_view.values.shape == torch.Size([1, 1])


def test_build_paired_swing_auxiliary_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_swing_auxiliary_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.5,
        margin=0.25,
        positive_action_source="teacher_action",
        negative_action_source="actions",
        loss_scope="label_mean",
        compare_to=" Top_Other ",
        margin_retention_coef=0.1,
        margin_retention_margin=0.2,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        swing_metrics={"paired_swing_rows": 2.0, "paired_swing_weighted_loss": 99.0},
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_swing_weighted_loss"] == 99.0
    assert metrics["paired_swing_coef"] == pytest.approx(0.5)
    assert metrics["paired_swing_margin"] == pytest.approx(0.25)
    assert metrics["paired_swing_positive_action_source_teacher"] == 1.0
    assert metrics["paired_swing_negative_action_source_teacher"] == 0.0
    assert metrics["paired_swing_loss_scope_label_mean"] == 1.0
    assert metrics["paired_swing_compare_to_top_other"] == 1.0
    assert metrics["paired_swing_margin_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_swing_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_swing_rows"] == 2.0


def test_compute_paired_outcome_candidate_logps_preserves_current_reference_views() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": _packed_meta_from_ids(action_catalog, packed_ids),
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }
    inputs = resolve_paired_auxiliary_batch_inputs(
        learner,
        batch,
        packed_legal_error="paired outcome preference replay requires packed legal_ids/legal_offsets",
    )
    actions = learner._require_actions(batch["actions"], expected_shape=inputs.expected_shape)

    candidate_logps = compute_paired_outcome_candidate_logps(
        learner,
        batch,
        obs=inputs.obs,
        packed_legal=inputs.packed_legal,
        actions=actions,
        reset_before_step=None,
    )

    assert candidate_logps.current_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_action_logp.shape == torch.Size([2, 1])
    assert candidate_logps.current_best_non_target_logp.shape == torch.Size([2, 1])
    assert candidate_logps.reference_best_non_target_logp.shape == torch.Size([2, 1])
    assert torch.allclose(candidate_logps.current_action_logp, candidate_logps.reference_action_logp)
    assert torch.allclose(candidate_logps.current_best_non_target_logp, candidate_logps.reference_best_non_target_logp)
    assert candidate_logps.current_action_logp[0, 0] > candidate_logps.current_best_non_target_logp[0, 0]
    assert candidate_logps.current_action_logp[1, 0] < candidate_logps.current_best_non_target_logp[1, 0]
    assert model.factorized_candidate_logp_calls == 1


def test_build_paired_outcome_preference_context_preserves_detached_logp_surface() -> None:
    current_candidates = torch.tensor([0.0, 1.0], requires_grad=True)
    reference_candidates = torch.tensor([1.0, 0.0], requires_grad=True)
    candidate_logps = PairedOutcomeCandidateLogps(
        current_candidate_log_probs=current_candidates,
        reference_candidate_log_probs=reference_candidates,
        current_action_logp=torch.tensor([[0.1]], requires_grad=True),
        current_best_non_target_logp=torch.tensor([[0.2]], requires_grad=True),
        reference_action_logp=torch.tensor([[0.3]], requires_grad=True),
        reference_best_non_target_logp=torch.tensor([[0.4]], requires_grad=True),
    )

    context = build_paired_outcome_preference_context(
        weighted_loss=torch.tensor(0.75, requires_grad=True),
        loss_mask=torch.tensor([[1.0]], requires_grad=True),
        candidate_logps=candidate_logps,
        preference_context={"paired_outcome_preference_margins": torch.tensor([0.5])},
    )

    assert context["paired_outcome_preference_loss"].item() == pytest.approx(0.75)
    assert context["policy_train_mask"].tolist() == [[1.0]]
    assert context["current_action_logp"].reshape(-1).tolist() == pytest.approx([0.1])
    assert context["current_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.2])
    assert context["reference_action_logp"].reshape(-1).tolist() == pytest.approx([0.3])
    assert context["reference_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.4])
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.5])
    assert not context["paired_outcome_preference_loss"].requires_grad
    assert not context["policy_train_mask"].requires_grad
    assert not context["current_action_logp"].requires_grad


def test_build_paired_outcome_preference_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_outcome_preference_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.7,
        beta=0.2,
        aggregation=" Sum ",
        group_balance=True,
        retention_coef=0.1,
        retention_margin=0.2,
        retention_reference_top_only=True,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        top_action_retention_reference_top_only=True,
        preference_metrics={
            "paired_outcome_preference_pair_count": 2.0,
            "paired_outcome_preference_weighted_loss": 99.0,
        },
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_outcome_preference_weighted_loss"] == 99.0
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_outcome_preference_retention_margin"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_coef"] == pytest.approx(0.3)
    assert metrics["paired_outcome_preference_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_outcome_preference_top_action_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 2.0


def test_impala_learner_paired_outcome_auxiliary_preserves_factorized_metrics() -> None:
    action_catalog = _teacher_aux_catalog()
    model = FactorizedStructuredTeacherModel(action_catalog)
    learner = ImpalaLearner(model=model, pass_action_id=action_catalog.pass_action_id)
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.25]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "preference_pair_id": np.asarray([[7], [7]], dtype=np.int64),
        "preference_role": np.asarray([[1], [0]], dtype=np.int64),
        "preference_group_id": np.asarray([[3], [3]], dtype=np.int64),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    loss, metrics, context = learner._paired_outcome_preference_loss_and_metrics(
        batch,
        beta=0.2,
        coef=0.7,
        aggregation="sum",
        group_balance=True,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(metrics["paired_outcome_preference_weighted_loss"])
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 1.0
    assert metrics["paired_outcome_preference_group_count"] == 1.0
    assert context["current_action_logp"].shape == (2, 1)
    assert context["reference_action_logp"].shape == (2, 1)
    assert context["current_best_non_target_logp"].shape == (2, 1)
    assert context["reference_best_non_target_logp"].shape == (2, 1)
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.0])
    assert model.factorized_candidate_logp_calls == 1


def test_impala_learner_auxiliary_update_averages_multiple_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 3
    assert model.public_target_profiles == ["base", "aggressive", "control"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_auxiliary_update_cycles_public_heuristic_profiles() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["aggressive"]
    assert metrics["teacher_public_heuristic_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_public_heuristic_target_entropy"] > 0.0


def test_impala_learner_public_heuristic_profiles_fall_back_to_base_after_end_update() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_temperature=1.0,
        teacher_public_heuristic_profiles=("base", "aggressive", "control"),
        teacher_public_heuristic_profile_mode="cycle",
        teacher_public_heuristic_profiles_end_updates=0,
    )
    learner.update_count = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = np.asarray([0, 5, action_catalog.pass_action_id], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.public_target_calls == 1
    assert model.public_target_profiles == ["base"]


def test_impala_learner_auxiliary_update_optimizes_teacher_only_loss() -> None:
    torch.manual_seed(0)

    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((2, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    legal_mask[1, 0, [10, 11, 12, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_action": np.asarray([[0], [11]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] > 0.0
    assert metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_auxiliary_update_handles_batches_without_valid_teacher_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
        teacher_action_coef=0.2,
    )
    legal_mask = np.zeros((1, 1, action_catalog.action_space_size), dtype=np.uint8)
    legal_mask[0, 0, [0, 5, 19]] = 1
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    batch = {
        "obs": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "legal_mask": legal_mask,
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[False]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0


def test_impala_learner_auxiliary_update_uses_factorized_teacher_path() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=FactorizedStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.5,
        teacher_slot_coef=0.25,
        teacher_attack_type_coef=0.1,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(action_catalog.attack_type_names)}
    packed_ids = np.asarray([0, 5, 19, 10, 11, 12, 19], dtype=np.uint32)
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": np.asarray([[0], [1]], dtype=np.int64),
        "initial_hidden_state": np.zeros((1, 2, 1), dtype=np.float32),
        "teacher_family": np.asarray(
            [[family_index["main_play_character"]], [family_index["attack"]]],
            dtype=np.int64,
        ),
        "teacher_slot": np.asarray([[0], [0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1], [attack_type_index["direct"]]], dtype=np.int64),
        "teacher_valid": np.asarray([[True], [True]], dtype=np.bool_),
        "policy_train_mask": np.asarray([[True], [True]], dtype=np.bool_),
    }

    metrics = learner.auxiliary_update(batch)

    model = learner.model
    assert isinstance(model, FactorizedStructuredTeacherModel)
    assert model.factorized_calls == 1
    assert metrics["loss"] > 0.0
    assert metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_slot_accuracy"] == pytest.approx(1.0)
    assert metrics["teacher_attack_type_accuracy"] == pytest.approx(1.0)
