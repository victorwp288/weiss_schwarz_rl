"""Train a simulator-facing frozen-B1 residual policy on counterfactual labels.

Unlike ``b1_residual_adoption_probe.py``, this script trains through the same
live frozen-B1 wrapper used in closed-loop eval. The base B1 model stays frozen;
only the residual head receives gradients. The output residual_state.pt remains
compatible with ``b1_residual_closed_loop_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import b1_counterfactual_labels as cf
import torch
import torch.nn.functional as F
from torch import Tensor
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.residual_policy import FrozenStoredLogitResidual, TrainableLiveFrozenB1Residual


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: Any) -> int:
    return int(torch.as_tensor(value).item())


def _action_family_ids(*, action_dim: int, catalog: ActionCatalog | None) -> tuple[Tensor | None, tuple[str, ...]]:
    if catalog is None:
        return None, ()
    family_names = tuple(family.name for family in catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    ids = torch.full((int(action_dim),), -1, dtype=torch.long)
    for action_id in range(int(action_dim)):
        try:
            decoded = catalog.decode(action_id)
        except Exception:
            continue
        ids[action_id] = int(family_index.get(decoded.family, -1))
    return ids, family_names


def _label_index(legal_ids: Tensor, action_id: int) -> int:
    matches = torch.nonzero(legal_ids.to(dtype=torch.long) == int(action_id), as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise RuntimeError(f"action_id={action_id} appears {int(matches.numel())} times in legal ids")
    return int(matches[0].item())


def _load_records(label_dirs: Sequence[Path], *, max_labels: int) -> list[dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    for source_dir in label_dirs:
        for row in _read_jsonl(source_dir / "counterfactual_labels.jsonl"):
            enriched = dict(row)
            enriched["_source_label_dir"] = source_dir.as_posix()
            label_rows.append(enriched)
    if int(max_labels) > 0:
        label_rows = label_rows[: int(max_labels)]
    records: list[dict[str, Any]] = []
    for row in label_rows:
        ref = row.get("tensor_ref")
        if not isinstance(ref, str) or not ref:
            raise SystemExit(f"label {row.get('label_id')} is missing tensor_ref")
        source_dir = Path(str(row["_source_label_dir"]))
        loaded = torch.load(source_dir / ref, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise SystemExit(f"tensor_ref did not load to a dict: {source_dir / ref}")
        loaded["_source_label_dir"] = source_dir.as_posix()
        records.append(loaded)
    return records


def _split_records(
    records: Sequence[dict[str, Any]], *, validation_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fraction = float(validation_fraction)
    if fraction <= 0.0 or len(records) < 2:
        return list(records), []
    stride = max(2, round(1.0 / min(max(fraction, 1e-6), 0.9)))
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index % stride == stride - 1:
            validation.append(record)
        else:
            train.append(record)
    if not train:
        train, validation = list(records[:1]), list(records[1:])
    return train, validation


def _record_family(record: Mapping[str, Any]) -> str:
    for key in ("positive_family", "action_family", "selected_family"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    action_label = record.get("action_label")
    if isinstance(action_label, Mapping):
        value = action_label.get("family")
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _base_param_l2(model: torch.nn.Module, baseline: Mapping[str, Tensor]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        before = baseline.get(name)
        if before is None:
            continue
        delta = parameter.detach().cpu() - before
        total += float(torch.sum(delta * delta).item())
    return math.sqrt(total)


def _live_logits(
    wrapper: TrainableLiveFrozenB1Residual,
    record: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    obs = torch.as_tensor(record["obs"], dtype=torch.float32, device=device).reshape(1, -1)
    actor = torch.as_tensor(record["actor_seat"], dtype=torch.long, device=device).reshape(1)
    legal_ids = torch.as_tensor(record["legal_ids"], dtype=torch.long, device=device)
    hidden = wrapper.initial_seat_hidden(1, device=device, dtype=obs.dtype)
    logits, _value, _next_hidden = wrapper.forward_seat_aware(obs, actor, hidden, scoring_mode="learner")
    legal_logits = logits[0].index_select(0, legal_ids)
    residual_full = wrapper.residual_probe.residual_logits(obs, actor)[0]
    residual_legal = residual_full.index_select(0, legal_ids)
    return legal_logits, residual_legal, legal_ids, logits[0]


def _metric_for_record(
    wrapper: TrainableLiveFrozenB1Residual,
    record: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    legal_logits, residual_legal, legal_ids, _full_logits = _live_logits(wrapper, record, device=device)
    positive_id = _as_int(record["positive_action_id"])
    baseline_id = _as_int(record["baseline_action_id"])
    pos_index = _label_index(legal_ids, positive_id)
    base_index = _label_index(legal_ids, baseline_id)
    probs = F.softmax(legal_logits, dim=-1)
    stored_base = torch.as_tensor(record.get("base_s1_legal_logits", []), dtype=torch.float32, device=device)
    stored_diff = None
    if stored_base.numel() == legal_logits.numel():
        stored_diff = float(torch.max(torch.abs(legal_logits.detach() - stored_base)).detach().cpu().item())
    return {
        "positive_probability": float(probs[pos_index].detach().cpu().item()),
        "baseline_probability": float(probs[base_index].detach().cpu().item()),
        "positive_minus_baseline_logit_margin": float(
            (legal_logits[pos_index] - legal_logits[base_index]).detach().cpu().item()
        ),
        "top_action_id": int(legal_ids[int(torch.argmax(legal_logits).item())].detach().cpu().item()),
        "residual_l2_norm_legal": float(torch.linalg.vector_norm(residual_legal.detach()).cpu().item()),
        "residual_linf_norm_legal": float(torch.max(torch.abs(residual_legal.detach())).cpu().item()),
        "max_abs_diff_vs_stored_b1_s1_legal_logits": stored_diff,
    }


def _train_step(
    wrapper: TrainableLiveFrozenB1Residual,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    margin: float,
    residual_l2_coef: float,
) -> Tensor:
    legal_logits, residual_legal, legal_ids, _full_logits = _live_logits(wrapper, record, device=device)
    positive_id = _as_int(record["positive_action_id"])
    baseline_id = _as_int(record["baseline_action_id"])
    pos_index = _label_index(legal_ids, positive_id)
    base_index = _label_index(legal_ids, baseline_id)
    label_weight = float(torch.as_tensor(record.get("label_weight", 1.0)).item())
    ce = -F.log_softmax(legal_logits, dim=-1)[pos_index] * label_weight
    margin_loss = F.relu(
        torch.tensor(float(margin), device=device) - legal_logits[pos_index] + legal_logits[base_index]
    )
    residual_l2 = torch.mean(residual_legal * residual_legal)
    return ce + 0.2 * margin_loss * label_weight + float(residual_l2_coef) * residual_l2


def _adoption_summary(records: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adopted = 0
    by_family: dict[str, dict[str, int]] = {}
    for record, item in zip(records, metrics, strict=True):
        positive_id = _as_int(record["positive_action_id"])
        is_adopted = int(item["top_action_id"]) == positive_id
        adopted += int(is_adopted)
        family = _record_family(record)
        bucket = by_family.setdefault(family, {"count": 0, "adopted": 0})
        bucket["count"] += 1
        bucket["adopted"] += int(is_adopted)
    return {
        "count": int(len(records)),
        "adoption_count": int(adopted),
        "adoption_rate": float(adopted / max(len(records), 1)),
        "by_positive_family": {
            family: {
                "count": values["count"],
                "adoption_count": values["adopted"],
                "adoption_rate": float(values["adopted"] / max(values["count"], 1)),
            }
            for family, values in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b1-baseline-run-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--residual-l2-coef", type=float, default=1e-5)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--residual-mode", choices=("plain", "gated", "family_gated"), default="plain")
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-adoption-rate",
        type=float,
        default=0.0,
        help="Stop once train-label adoption reaches this rate and the mean positive probability target is met.",
    )
    parser.add_argument(
        "--early-stop-mean-positive-prob",
        type=float,
        default=0.0,
        help="Companion target for --early-stop-adoption-rate. Use this to avoid over-driving sparse labels.",
    )
    parser.add_argument("--early-stop-check-every", type=int, default=25)
    args = parser.parse_args()

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    label_dirs = [Path(item) for item in args.label_dir]
    records = _load_records(label_dirs, max_labels=int(args.max_labels))
    if not records:
        raise SystemExit("no tensorized labels loaded")
    obs_dim = int(torch.as_tensor(records[0]["obs"]).numel())

    ctx = cf._build_in_process_context(
        stack_config=args.stack_config,
        run_dir=args.run_dir,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
        checkpoint_policies=(),
        pairs=1,
        seed_scope="b1_trainable_residual_policy",
        device=str(device),
        public_heuristic_bias_scale=1.0,
        include_builtin=(),
    )
    base_policy = ctx.policies[cf.matrix.NO_LEAGUE_POLICY_ID]
    if base_policy.model is None:
        raise RuntimeError("B1 baseline resolved without a model")
    action_family_ids, family_names = _action_family_ids(action_dim=ctx.action_dim, catalog=ctx.action_catalog)
    residual = FrozenStoredLogitResidual(
        obs_dim=obs_dim,
        action_dim=ctx.action_dim,
        hidden_dim=int(args.hidden_dim),
        alpha=float(args.alpha),
        residual_mode=str(args.residual_mode),
        action_family_ids=action_family_ids,
        family_count=len(family_names),
        gate_bias=float(args.gate_bias),
    ).to(device)
    wrapper = TrainableLiveFrozenB1Residual(base_model=base_policy.model, residual_probe=residual).to(device)
    wrapper.train()
    wrapper.base_model.eval()
    base_before = {name: parameter.detach().cpu().clone() for name, parameter in wrapper.base_model.named_parameters()}
    optimizer = torch.optim.AdamW(wrapper.residual_probe.parameters(), lr=float(args.lr))
    train_records, validation_records = _split_records(records, validation_fraction=float(args.validation_fraction))

    with torch.no_grad():
        before_metrics = [_metric_for_record(wrapper, record, device=device) for record in records]
        live_identity_max_abs_diff = 0.0
        stored_diff_values: list[float] = []
        for item in before_metrics:
            diff = item.get("max_abs_diff_vs_stored_b1_s1_legal_logits")
            if diff is not None:
                stored_diff_values.append(float(diff))
            live_identity_max_abs_diff = max(live_identity_max_abs_diff, float(diff or 0.0))

    losses: list[float] = []
    stopped_at_step: int | None = None
    for step in range(int(args.steps)):
        total_loss = torch.zeros((), device=device)
        for record in train_records:
            total_loss = total_loss + _train_step(
                wrapper,
                record,
                device=device,
                margin=float(args.margin),
                residual_l2_coef=float(args.residual_l2_coef),
            )
        total_loss = total_loss / max(len(train_records), 1)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))
        if (
            float(args.early_stop_adoption_rate) > 0.0
            and int(args.early_stop_check_every) > 0
            and (step + 1) % int(args.early_stop_check_every) == 0
        ):
            with torch.no_grad():
                train_metrics_now = [_metric_for_record(wrapper, record, device=device) for record in train_records]
            train_adoption_now = _adoption_summary(train_records, train_metrics_now)
            mean_positive_prob = sum(float(item["positive_probability"]) for item in train_metrics_now) / max(
                len(train_metrics_now),
                1,
            )
            if float(train_adoption_now["adoption_rate"]) >= float(
                args.early_stop_adoption_rate
            ) and mean_positive_prob >= float(args.early_stop_mean_positive_prob):
                stopped_at_step = step + 1
                break

    with torch.no_grad():
        after_metrics = [_metric_for_record(wrapper, record, device=device) for record in records]
    train_ids = {id(record) for record in train_records}
    validation_ids = {id(record) for record in validation_records}
    train_after = [metric for record, metric in zip(records, after_metrics, strict=True) if id(record) in train_ids]
    validation_after = [
        metric for record, metric in zip(records, after_metrics, strict=True) if id(record) in validation_ids
    ]
    adoption = _adoption_summary(records, after_metrics)
    train_adoption = _adoption_summary(train_records, train_after)
    validation_adoption = _adoption_summary(validation_records, validation_after) if validation_records else None
    base_delta_l2 = _base_param_l2(wrapper.base_model, base_before)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual_path = args.output_dir / "residual_state.pt"
    torch.save(
        {
            "format": "b1_frozen_stored_logit_residual_v1",
            "training_format": "b1_trainable_live_frozen_residual_v1",
            "model_state_dict": wrapper.residual_probe.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": int(ctx.action_dim),
            "hidden_dim": int(args.hidden_dim),
            "alpha": float(args.alpha),
            "residual_mode": str(args.residual_mode),
            "action_family_ids": None if action_family_ids is None else action_family_ids.cpu(),
            "family_count": len(family_names),
            "family_names": family_names,
            "gate_bias": float(args.gate_bias),
        },
        residual_path,
    )
    report = {
        "format": "b1_trainable_residual_policy_report_v1",
        "label_dirs": [path.as_posix() for path in label_dirs],
        "label_count": len(records),
        "train_label_count": len(train_records),
        "validation_label_count": len(validation_records),
        "device": str(device),
        "obs_dim": obs_dim,
        "action_dim": int(ctx.action_dim),
        "alpha": float(args.alpha),
        "residual_mode": str(args.residual_mode),
        "gate_bias": float(args.gate_bias),
        "family_names": list(family_names),
        "steps": int(args.steps),
        "stopped_at_step": stopped_at_step,
        "early_stop_adoption_rate": float(args.early_stop_adoption_rate),
        "early_stop_mean_positive_prob": float(args.early_stop_mean_positive_prob),
        "lr": float(args.lr),
        "margin": float(args.margin),
        "residual_l2_coef": float(args.residual_l2_coef),
        "base_param_delta_l2": float(base_delta_l2),
        "base_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in wrapper.base_model.parameters() if parameter.requires_grad)
        ),
        "residual_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in wrapper.residual_probe.parameters() if parameter.requires_grad)
        ),
        "train_loss_initial": losses[0] if losses else math.nan,
        "train_loss_final": losses[-1] if losses else math.nan,
        "live_identity_max_abs_diff_vs_stored_b1_s1_at_init": float(live_identity_max_abs_diff),
        "live_vs_stored_b1_s1_max_abs_diff_mean_at_init": None
        if not stored_diff_values
        else float(sum(stored_diff_values) / len(stored_diff_values)),
        "adoption_count": int(adoption["adoption_count"]),
        "adoption_rate": float(adoption["adoption_rate"]),
        "adoption_summary": adoption,
        "train_adoption_summary": train_adoption,
        "validation_adoption_summary": validation_adoption,
        "labels": [
            {
                "label_id": str(record.get("label_id")),
                "source_label_dir": str(record.get("_source_label_dir", "")),
                "positive_action_id": _as_int(record["positive_action_id"]),
                "baseline_action_id": _as_int(record["baseline_action_id"]),
                "before": before,
                "after": after,
            }
            for record, before, after in zip(records, before_metrics, after_metrics, strict=True)
        ],
        "residual_state_path": residual_path.relative_to(args.output_dir).as_posix(),
    }
    _write_json(args.output_dir / "trainable_residual_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
