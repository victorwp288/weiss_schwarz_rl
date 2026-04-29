"""Train a minimal frozen-base residual on tensorized B1 counterfactual labels.

This is a supervised adoption proof, not a league/exploiter trainer. The base
policy is represented by stored B1/S1 legal logits captured at the labelled
decision; the optimizer only updates the residual head.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.residual_policy import FrozenStoredLogitResidual


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _infer_run_dir(label_dir: Path) -> Path | None:
    resolved = label_dir.resolve()
    for parent in [resolved, *resolved.parents]:
        if (parent / "spec_bundle.json").is_file():
            return parent
    return None


def _infer_action_dim(label_dir: Path, records: Sequence[Mapping[str, Any]], explicit: int | None) -> int:
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    run_dir = _infer_run_dir(label_dir)
    if run_dir is not None:
        spec_path = run_dir / "spec_bundle.json"
        if spec_path.is_file():
            spec = _read_json(spec_path)
            action = spec.get("action", {}) if isinstance(spec, Mapping) else {}
            value = action.get("action_space_size") if isinstance(action, Mapping) else None
            if value is not None:
                return int(value)
    max_id = 0
    for record in records:
        legal_ids = torch.as_tensor(record["legal_ids"], dtype=torch.long)
        if legal_ids.numel():
            max_id = max(max_id, int(legal_ids.max().item()))
        for key in ("positive_action_id", "baseline_action_id"):
            value = record.get(key)
            if value is not None:
                max_id = max(max_id, int(torch.as_tensor(value).item()))
    return int(max_id + 1)


def _load_action_catalog(label_dir: Path) -> ActionCatalog | None:
    run_dir = _infer_run_dir(label_dir)
    if run_dir is None:
        return None
    spec_path = run_dir / "spec_bundle.json"
    if not spec_path.is_file():
        return None
    try:
        return ActionCatalog.from_spec_bundle(_read_json(spec_path))
    except Exception:
        return None


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


def _prob_and_top(model: FrozenStoredLogitResidual, record: Mapping[str, Any]) -> dict[str, Any]:
    legal_logits, base_logits, residual_legal = model.legal_logits(record)
    legal_ids = torch.as_tensor(record["legal_ids"], dtype=torch.long, device=legal_logits.device)
    positive_id = int(torch.as_tensor(record["positive_action_id"]).item())
    baseline_id = int(torch.as_tensor(record["baseline_action_id"]).item())
    pos_index = _label_index(legal_ids, positive_id)
    base_index = _label_index(legal_ids, baseline_id)
    probs = F.softmax(legal_logits, dim=-1)
    base_probs = F.softmax(base_logits, dim=-1)
    obs = torch.as_tensor(record["obs"], dtype=torch.float32, device=legal_logits.device).reshape(1, -1)
    actor_seat = torch.as_tensor(record["actor_seat"], dtype=torch.long, device=legal_logits.device).reshape(1)
    gate = model.gate_values(obs, actor_seat, legal_ids)
    return {
        "positive_probability": float(probs[pos_index].detach().cpu().item()),
        "baseline_probability": float(probs[base_index].detach().cpu().item()),
        "base_positive_probability": float(base_probs[pos_index].detach().cpu().item()),
        "base_baseline_probability": float(base_probs[base_index].detach().cpu().item()),
        "positive_minus_baseline_logit_margin": float(
            (legal_logits[pos_index] - legal_logits[base_index]).detach().cpu().item()
        ),
        "base_positive_minus_baseline_logit_margin": float(
            (base_logits[pos_index] - base_logits[base_index]).detach().cpu().item()
        ),
        "top_action_id": int(legal_ids[int(torch.argmax(legal_logits).item())].detach().cpu().item()),
        "base_top_action_id": int(legal_ids[int(torch.argmax(base_logits).item())].detach().cpu().item()),
        "residual_l2_norm_legal": float(torch.linalg.vector_norm(residual_legal.detach()).cpu().item()),
        "residual_linf_norm_legal": float(torch.max(torch.abs(residual_legal.detach())).cpu().item()),
        "gate_mean_legal": None if gate is None else float(torch.mean(gate.detach()).cpu().item()),
        "gate_min_legal": None if gate is None else float(torch.min(gate.detach()).cpu().item()),
        "gate_max_legal": None if gate is None else float(torch.max(gate.detach()).cpu().item()),
    }


def _train_step(
    model: FrozenStoredLogitResidual,
    record: Mapping[str, Any],
    *,
    margin: float,
    residual_l2_coef: float,
) -> Tensor:
    legal_logits, _base_logits, residual_legal = model.legal_logits(record)
    legal_ids = torch.as_tensor(record["legal_ids"], dtype=torch.long, device=legal_logits.device)
    positive_id = int(torch.as_tensor(record["positive_action_id"]).item())
    baseline_id = int(torch.as_tensor(record["baseline_action_id"]).item())
    pos_index = _label_index(legal_ids, positive_id)
    base_index = _label_index(legal_ids, baseline_id)
    label_weight = float(torch.as_tensor(record.get("label_weight", 1.0)).item())
    ce = -F.log_softmax(legal_logits, dim=-1)[pos_index] * label_weight
    margin_loss = F.relu(
        torch.tensor(float(margin), device=legal_logits.device) - legal_logits[pos_index] + legal_logits[base_index]
    )
    residual_l2 = torch.mean(residual_legal * residual_legal)
    return ce + 0.2 * margin_loss * label_weight + float(residual_l2_coef) * residual_l2


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


def _adoption_summary(records: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adopted = 0
    by_family: dict[str, dict[str, int]] = {}
    for record, item in zip(records, metrics, strict=True):
        positive_id = int(torch.as_tensor(record["positive_action_id"]).item())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing counterfactual_labels.jsonl; may be repeated for multi-label training.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--residual-l2-coef", type=float, default=1e-5)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--residual-mode", choices=("plain", "gated", "family_gated"), default="plain")
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    args = parser.parse_args()

    label_dirs = [Path(item) for item in args.label_dir]
    label_dir = label_dirs[0]
    output_dir = args.output_dir or (label_dir / "residual_adoption_probe")
    label_rows: list[dict[str, Any]] = []
    for source_dir in label_dirs:
        for row in _read_jsonl(source_dir / "counterfactual_labels.jsonl"):
            enriched = dict(row)
            enriched["_source_label_dir"] = source_dir.as_posix()
            label_rows.append(enriched)
    if not label_rows:
        raise SystemExit(f"no labels found in {[str(path / 'counterfactual_labels.jsonl') for path in label_dirs]}")
    if int(args.max_labels) > 0:
        label_rows = label_rows[: int(args.max_labels)]

    records: list[dict[str, Any]] = []
    for row in label_rows:
        ref = row.get("tensor_ref")
        if not isinstance(ref, str) or not ref:
            raise SystemExit(f"label {row.get('label_id')} is missing tensor_ref; rerun counterfactual label capture")
        source_dir = Path(str(row.get("_source_label_dir", label_dir.as_posix())))
        record = torch.load(source_dir / ref, map_location="cpu", weights_only=False)
        if not isinstance(record, dict):
            raise SystemExit(f"tensor_ref did not load to a dict: {ref}")
        record["_source_label_dir"] = source_dir.as_posix()
        records.append(record)

    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    obs_dim = int(torch.as_tensor(records[0]["obs"]).numel())
    action_dim = _infer_action_dim(label_dir, records, args.action_dim)
    action_family_ids, family_names = _action_family_ids(
        action_dim=action_dim,
        catalog=_load_action_catalog(label_dir),
    )
    model = FrozenStoredLogitResidual(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(args.hidden_dim),
        alpha=float(args.alpha),
        residual_mode=str(args.residual_mode),
        action_family_ids=action_family_ids,
        family_count=len(family_names),
        gate_bias=float(args.gate_bias),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    train_records, validation_records = _split_records(records, validation_fraction=float(args.validation_fraction))

    with torch.no_grad():
        before_metrics = [_prob_and_top(model, record) for record in records]
        identity_max_abs_diff = 0.0
        for record in records:
            legal_logits, base_logits, _residual = model.legal_logits(record)
            identity_max_abs_diff = max(
                identity_max_abs_diff,
                float(torch.max(torch.abs(legal_logits - base_logits)).detach().cpu().item()),
            )

    losses: list[float] = []
    for _step in range(int(args.steps)):
        total_loss = torch.zeros((), device=device)
        for record in train_records:
            total_loss = total_loss + _train_step(
                model,
                record,
                margin=float(args.margin),
                residual_l2_coef=float(args.residual_l2_coef),
            )
        total_loss = total_loss / max(len(train_records), 1)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))

    with torch.no_grad():
        after_metrics = [_prob_and_top(model, record) for record in records]
    adoption = _adoption_summary(records, after_metrics)
    train_indices = {id(record) for record in train_records}
    validation_indices = {id(record) for record in validation_records}
    train_after = [metric for record, metric in zip(records, after_metrics, strict=True) if id(record) in train_indices]
    validation_after = [
        metric for record, metric in zip(records, after_metrics, strict=True) if id(record) in validation_indices
    ]
    train_adoption = _adoption_summary(train_records, train_after)
    validation_adoption = _adoption_summary(validation_records, validation_after) if validation_records else None

    residual_path = output_dir / "residual_state.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "b1_frozen_stored_logit_residual_v1",
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
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
        "format": "b1_residual_adoption_probe_report_v1",
        "label_dir": label_dir.as_posix(),
        "label_dirs": [path.as_posix() for path in label_dirs],
        "label_count": len(records),
        "device": str(device),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "alpha": float(args.alpha),
        "residual_mode": str(args.residual_mode),
        "gate_bias": float(args.gate_bias),
        "family_names": list(family_names),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "margin": float(args.margin),
        "residual_l2_coef": float(args.residual_l2_coef),
        "residual_zero_logit_max_abs_diff_vs_stored_b1_s1": float(identity_max_abs_diff),
        "base_trainable_parameter_count": 0,
        "base_param_delta_l2": 0.0,
        "train_loss_initial": losses[0] if losses else math.nan,
        "train_loss_final": losses[-1] if losses else math.nan,
        "train_label_count": len(train_records),
        "validation_label_count": len(validation_records),
        "adoption_count": int(adoption["adoption_count"]),
        "adoption_rate": float(adoption["adoption_rate"]),
        "adoption_summary": adoption,
        "train_adoption_summary": train_adoption,
        "validation_adoption_summary": validation_adoption,
        "labels": [
            {
                "label_id": str(record.get("label_id")),
                "source_label_dir": str(record.get("_source_label_dir", "")),
                "positive_action_id": int(torch.as_tensor(record["positive_action_id"]).item()),
                "baseline_action_id": int(torch.as_tensor(record["baseline_action_id"]).item()),
                "before": before,
                "after": after,
            }
            for record, before, after in zip(records, before_metrics, after_metrics, strict=True)
        ],
        "residual_state_path": residual_path.relative_to(output_dir).as_posix(),
    }
    _write_json(output_dir / "residual_adoption_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
