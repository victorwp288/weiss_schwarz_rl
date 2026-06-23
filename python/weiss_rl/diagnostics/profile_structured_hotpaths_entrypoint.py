from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.diagnostics.profile_fixtures import (
    empty_profile_observation,
    heuristic_profile_spec_bundle,
    packed_profile_meta,
    set_profile_stage,
    structured_profile_model_config,
    structured_profile_spec_bundle,
    typed_profile_observation_spec,
)
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.model import StructuredLegalPolicyValueModel, build_policy_value_model


def _build_heuristic_batch(
    *, rows: int
) -> tuple[HeuristicPublicPolicy, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    policy = HeuristicPublicPolicy.from_spec_bundle(heuristic_profile_spec_bundle())
    obs_rows = np.stack([empty_profile_observation() for _ in range(rows)], axis=0)
    row_legal_ids: list[np.ndarray] = []
    for row_index in range(rows):
        if row_index % 3 == 0:
            set_profile_stage(
                obs_rows[row_index],
                player_index=0,
                slot=0,
                occupied=True,
                power=5000,
                effective_soul=1,
            )
            row_legal_ids.append(np.array([472, 473, 474, 51, 402], dtype=np.uint32))
        elif row_index % 3 == 1:
            set_profile_stage(obs_rows[row_index], player_index=0, slot=1, occupied=True, power=2500)
            row_legal_ids.append(np.array([102, 103, 104, 105, 106], dtype=np.uint32))
        else:
            obs_rows[row_index][16] = 0
            obs_rows[row_index][17] = 3
            obs_rows[row_index][14] = 16
            obs_rows[row_index][15] = 40
            row_legal_ids.append(np.array([52, 53, 524, 525, 51], dtype=np.uint32))
    legal_ids = np.concatenate(row_legal_ids, axis=0)
    offsets = np.zeros((rows + 1,), dtype=np.uint32)
    cursor = 0
    for row_index, row_ids in enumerate(row_legal_ids):
        offsets[row_index] = cursor
        cursor += int(row_ids.size)
    offsets[-1] = cursor
    return policy, obs_rows, row_legal_ids, legal_ids, offsets


def _run_heuristic_benchmark(*, rows: int, iterations: int) -> dict[str, object]:
    policy, obs_rows, row_legal_ids, legal_ids, offsets = _build_heuristic_batch(rows=rows)
    meta = packed_profile_meta(legal_ids)

    scalar_started = time.perf_counter()
    for _ in range(iterations):
        scalar_actions = np.asarray(
            [
                policy.choose_action_from_meta(
                    obs_rows[row_index],
                    row_legal_ids[row_index],
                    packed_profile_meta(row_legal_ids[row_index]),
                )
                for row_index in range(rows)
            ],
            dtype=np.int64,
        )
    scalar_seconds = time.perf_counter() - scalar_started

    batch_started = time.perf_counter()
    for _ in range(iterations):
        batch_actions = policy.choose_actions_from_meta_batch(obs_rows, legal_ids, offsets, meta)
    batch_seconds = time.perf_counter() - batch_started

    return {
        "mode": "heuristic",
        "rows": int(rows),
        "iterations": int(iterations),
        "scalar_ms_per_iter": float((scalar_seconds / max(iterations, 1)) * 1000.0),
        "batch_ms_per_iter": float((batch_seconds / max(iterations, 1)) * 1000.0),
        "speedup_x": float(scalar_seconds / max(batch_seconds, 1e-9)),
        "actions_match": bool(np.array_equal(scalar_actions, batch_actions)),
    }


def _structured_meta(catalog: ActionCatalog, action_ids: np.ndarray) -> np.ndarray:
    unused = np.iinfo(np.uint16).max
    meta = np.full((int(action_ids.shape[0]), 4), unused, dtype=np.uint16)
    family_index = {family.name: index for index, family in enumerate(catalog.families)}
    attack_type_index = {name: index for index, name in enumerate(catalog.attack_type_names)}
    for row_index, action_id in enumerate(np.asarray(action_ids, dtype=np.int64).tolist()):
        decoded = catalog.decode(int(action_id))
        meta[row_index, 0] = np.uint16(family_index[decoded.family])
        if decoded.hand_index is not None:
            meta[row_index, 1] = np.uint16(decoded.hand_index)
        if decoded.stage_slot is not None:
            meta[row_index, 2] = np.uint16(decoded.stage_slot)
        if decoded.from_slot is not None:
            meta[row_index, 1] = np.uint16(decoded.from_slot)
        if decoded.to_slot is not None:
            meta[row_index, 2] = np.uint16(decoded.to_slot)
        if decoded.slot is not None:
            meta[row_index, 1] = np.uint16(decoded.slot)
        if decoded.attack_type is not None:
            meta[row_index, 2] = np.uint16(attack_type_index[decoded.attack_type])
        if decoded.index is not None:
            meta[row_index, 1] = np.uint16(decoded.index)
    return meta


def _build_structured_fixture(
    *,
    time_steps: int,
    batch_size: int,
) -> tuple[StructuredLegalPolicyValueModel, torch.Tensor, torch.Tensor, torch.Tensor, LegalActionBatch]:
    spec_bundle = structured_profile_spec_bundle()
    model = build_policy_value_model(
        observation_dim=18,
        config=structured_profile_model_config(),
        action_dim=9,
        observation_spec=typed_profile_observation_spec(),
        spec_bundle=spec_bundle,
    )
    assert isinstance(model, StructuredLegalPolicyValueModel)
    model.eval()
    obs = torch.zeros((time_steps, batch_size, 18), dtype=torch.float32)
    acting_seat = torch.zeros((time_steps, batch_size), dtype=torch.long)
    seat_hidden = model.initial_seat_hidden(batch_size)
    row_count = int(time_steps * batch_size)
    action_ids = np.tile(np.array([0, 1, 3, 4], dtype=np.uint32), row_count)
    offsets = np.arange(0, action_ids.size + 1, 4, dtype=np.uint32)
    catalog = ActionCatalog.from_spec_bundle(spec_bundle)
    meta = _structured_meta(catalog, action_ids)
    legal_actions = LegalActionBatch.from_packed(action_ids, offsets, meta=meta, action_space=9)
    return model, obs, acting_seat, seat_hidden, legal_actions


def _run_structured_benchmark(
    *,
    time_steps: int,
    batch_size: int,
    iterations: int,
    compile_trunk: bool,
) -> dict[str, object]:
    model, obs, acting_seat, seat_hidden, legal_actions = _build_structured_fixture(
        time_steps=time_steps,
        batch_size=batch_size,
    )
    compile_error: str | None = None
    if compile_trunk:
        try:
            model.enable_trunk_compile(mode="reduce-overhead")
        except Exception as exc:
            compile_error = repr(exc)
    with torch.inference_mode():
        for _ in range(5):
            model.forward_sequence_packed_seat_aware(obs, acting_seat, seat_hidden, legal_actions=legal_actions)

        trunk_started = time.perf_counter()
        for _ in range(iterations):
            recurrent_flat, state_repr, observation_context, values, next_hidden = (
                model.forward_trunk_sequence_seat_aware(
                    obs,
                    acting_seat,
                    seat_hidden,
                )
            )
        trunk_seconds = time.perf_counter() - trunk_started

        scorer_started = time.perf_counter()
        for _ in range(iterations):
            model.score_packed_legal_candidates(
                recurrent_flat,
                obs.reshape(time_steps * batch_size, obs.shape[-1]),
                legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
            )
        scorer_seconds = time.perf_counter() - scorer_started

        full_started = time.perf_counter()
        for _ in range(iterations):
            packed_logits, packed_values, packed_hidden = model.forward_sequence_packed_seat_aware(
                obs,
                acting_seat,
                seat_hidden,
                legal_actions=legal_actions,
            )
        full_seconds = time.perf_counter() - full_started

    return {
        "mode": "structured",
        "time_steps": int(time_steps),
        "batch_size": int(batch_size),
        "rows": int(time_steps * batch_size),
        "candidates": int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0,
        "iterations": int(iterations),
        "compile_trunk": bool(compile_trunk),
        "compile_error": compile_error or getattr(model, "_trunk_compile_last_error", None),
        "trunk_ms_per_iter": float((trunk_seconds / max(iterations, 1)) * 1000.0),
        "scorer_ms_per_iter": float((scorer_seconds / max(iterations, 1)) * 1000.0),
        "full_ms_per_iter": float((full_seconds / max(iterations, 1)) * 1000.0),
        "values_shape": list(packed_values.shape),
        "packed_logits_count": int(packed_logits.shape[0]),
        "hidden_shape": list(packed_hidden.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench structured Weiss RL hot paths.")
    parser.add_argument("--mode", choices=("heuristic", "structured"), required=True)
    parser.add_argument("--rows", type=int, default=2048, help="Heuristic rows to score.")
    parser.add_argument("--time-steps", type=int, default=32, help="Structured benchmark time dimension.")
    parser.add_argument("--batch-size", type=int, default=64, help="Structured benchmark batch dimension.")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument(
        "--compile-trunk", action="store_true", help="Enable torch.compile on the structured trunk before benchmarking."
    )
    args = parser.parse_args()

    if args.mode == "heuristic":
        result = _run_heuristic_benchmark(rows=int(args.rows), iterations=int(args.iterations))
    else:
        result = _run_structured_benchmark(
            time_steps=int(args.time_steps),
            batch_size=int(args.batch_size),
            iterations=int(args.iterations),
            compile_trunk=bool(args.compile_trunk),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
