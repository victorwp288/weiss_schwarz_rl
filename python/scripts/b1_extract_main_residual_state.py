from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _infer_hidden_dim(state: Mapping[str, Any]) -> int:
    weight = state.get("residual.0.weight")
    if torch.is_tensor(weight) and weight.ndim == 2:
        return int(weight.shape[0])
    raise RuntimeError("could not infer residual hidden_dim from residual.0.weight")


def _infer_obs_dim(state: Mapping[str, Any]) -> int:
    weight = state.get("residual.0.weight")
    seat_weight = state.get("seat_embedding.weight")
    if torch.is_tensor(weight) and weight.ndim == 2:
        seat_dim = int(seat_weight.shape[1]) if torch.is_tensor(seat_weight) and seat_weight.ndim == 2 else 8
        return int(weight.shape[1]) - seat_dim
    raise RuntimeError("could not infer residual obs_dim from residual.0.weight")


def _infer_action_dim(state: Mapping[str, Any]) -> int:
    bias = state.get("residual.2.bias")
    if torch.is_tensor(bias) and bias.ndim == 1:
        return int(bias.shape[0])
    raise RuntimeError("could not infer residual action_dim from residual.2.bias")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a TrainableLiveFrozenB1Residual checkpoint into residual_state.pt format."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--residual-mode", default="plain")
    parser.add_argument("--gate-bias", type=float, default=0.0)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise RuntimeError(f"checkpoint missing model_state_dict: {args.checkpoint}")
    model_state = payload["model_state_dict"]
    residual_state: dict[str, Any] = {}
    prefix = "residual_probe."
    for key, value in model_state.items():
        if str(key).startswith(prefix):
            residual_state[str(key)[len(prefix) :]] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    if not residual_state:
        raise RuntimeError(f"checkpoint does not contain residual_probe.* parameters: {args.checkpoint}")
    output_payload = {
        "obs_dim": _infer_obs_dim(residual_state),
        "action_dim": _infer_action_dim(residual_state),
        "hidden_dim": _infer_hidden_dim(residual_state),
        "alpha": float(args.alpha),
        "residual_mode": str(args.residual_mode),
        "gate_bias": float(args.gate_bias),
        "model_state_dict": residual_state,
        "source_checkpoint": str(args.checkpoint),
        "source_update_count": int(payload.get("update_count", -1)),
        "format": "frozen_stored_logit_residual_state_v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    print(f"Wrote residual state: {args.output}")


if __name__ == "__main__":
    main()
