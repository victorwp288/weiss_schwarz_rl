from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.final.matchup_manifest import final_eval_matchup_manifest_rows
from weiss_rl.eval.final.matrix_artifacts import (
    legacy_payoff_matrix_name,
    write_final_eval_matrix_artifacts,
    write_matrix_csv,
)
from weiss_rl.eval.final.payload_sections import relative_to
from weiss_rl.eval.final.run_diagnostics import (
    build_run_level_seat_bias_payload,
    write_replay_diagnostics,
    write_run_level_diagnostics,
    write_truncation_heatmap_csv,
)
from weiss_rl.eval.simulator.harness import EvalGameRecord, write_episodes_jsonl


@dataclass(frozen=True, slots=True)
class FinalEvalArtifactWriteStep:
    step_id: str
    title: str
    purpose: str
    output_paths: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "purpose": self.purpose,
            "output_paths": list(self.output_paths),
        }


@dataclass(frozen=True, slots=True)
class FinalEvalArtifactPaths:
    metadata_json: Path
    policy_set_json: Path
    summary_json: Path
    posterior_samples_json: Path
    matchups_csv: Path
    episodes_jsonl: Path


FINAL_EVAL_ARTIFACT_WRITE_PLAN = (
    FinalEvalArtifactWriteStep(
        step_id="core_json",
        title="Core JSON artifacts",
        purpose="Write metadata, policy set, summary, and posterior samples.",
        output_paths=("metadata.json", "policy_set.json", "summary.json", "posterior_samples.json"),
    ),
    FinalEvalArtifactWriteStep(
        step_id="posterior_npz",
        title="Posterior NPZ",
        purpose="Write compressed posterior samples when using the canonical run layout.",
        output_paths=("posterior_samples.npz",),
    ),
    FinalEvalArtifactWriteStep(
        step_id="matrix_exports",
        title="Matrix exports",
        purpose="Write JSON and CSV matrices, including legacy payoff matrix names.",
        output_paths=("matrices/*.json", "matrices/*.csv", "matrices/mean.csv"),
    ),
    FinalEvalArtifactWriteStep(
        step_id="matchup_manifest",
        title="Matchup manifest",
        purpose="Write the canonical matchup manifest CSV.",
        output_paths=("matchups.csv",),
    ),
    FinalEvalArtifactWriteStep(
        step_id="aggregate_episodes",
        title="Aggregate episodes",
        purpose="Write aggregate episode records when matchup results include records.",
        output_paths=("episodes.jsonl",),
    ),
    FinalEvalArtifactWriteStep(
        step_id="canonical_sidecars",
        title="Canonical sidecars",
        purpose="Write run-level diagnostics, replay metadata, and artifact hashes for canonical run directories.",
        output_paths=(
            "eval/diagnostics/seat_bias.json",
            "eval/diagnostics/truncation_heatmap.csv",
            "eval/diagnostics/replay_verification.json",
            "replays/index.json",
            "eval/final_eval/artifact_hashes.json",
        ),
    ),
)


def final_eval_artifact_write_plan_payload() -> list[dict[str, object]]:
    return [step.as_payload() for step in FINAL_EVAL_ARTIFACT_WRITE_PLAN]


def write_final_eval_artifacts(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    layout = maybe_layout(output_dir)
    paths = final_eval_artifact_paths(output_dir=output_dir, layout=layout)

    write_core_final_eval_json_artifacts(paths=paths, payload=payload)
    write_final_eval_posterior_npz_if_canonical(layout=layout, payload=payload)
    matrices = cast(Mapping[str, Mapping[str, Any]], payload["matrices"])
    write_final_eval_matrix_artifacts(output_dir=output_dir, matrices=matrices, layout=layout)
    write_final_eval_matchup_manifest(paths=paths, output_dir=output_dir, matchup_results=matchup_results)
    write_aggregate_final_eval_episodes(paths=paths, matchup_results=matchup_results)
    if layout is not None:
        write_canonical_final_eval_sidecars(layout=layout, payload=payload, matchup_results=matchup_results)


def final_eval_artifact_paths(*, output_dir: Path, layout: ArtifactLayout | None) -> FinalEvalArtifactPaths:
    return FinalEvalArtifactPaths(
        metadata_json=layout.final_eval_metadata_json() if layout is not None else output_dir / "metadata.json",
        policy_set_json=layout.final_eval_policy_set_json() if layout is not None else output_dir / "policy_set.json",
        summary_json=layout.final_eval_summary_json() if layout is not None else output_dir / "summary.json",
        posterior_samples_json=(
            layout.final_eval_posterior_samples_json() if layout is not None else output_dir / "posterior_samples.json"
        ),
        matchups_csv=layout.final_eval_matchups_csv() if layout is not None else output_dir / "matchups.csv",
        episodes_jsonl=layout.final_eval_episodes_jsonl() if layout is not None else output_dir / "episodes.jsonl",
    )


def write_core_final_eval_json_artifacts(*, paths: FinalEvalArtifactPaths, payload: Mapping[str, Any]) -> None:
    paths.metadata_json.write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.policy_set_json.write_text(
        json.dumps({"policy_ids": payload["policy_ids"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths.posterior_samples_json.write_text(
        json.dumps(payload["posterior_samples"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_final_eval_posterior_npz_if_canonical(*, layout: ArtifactLayout | None, payload: Mapping[str, Any]) -> None:
    if layout is not None:
        posterior_payload = cast(Mapping[str, Any], payload["posterior_samples"])
        np.savez_compressed(
            layout.final_eval_posterior_samples_npz(),
            values=np.asarray(posterior_payload.get("values", ()), dtype=np.float64),
            policy_ids=np.asarray(posterior_payload.get("policy_ids", ()), dtype=object),
        )


def write_final_eval_matchup_manifest(
    *,
    paths: FinalEvalArtifactPaths,
    output_dir: Path,
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    manifest_rows = final_eval_matchup_manifest_rows(output_dir=output_dir, matchup_results=matchup_results)
    with paths.matchups_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_aggregate_final_eval_episodes(
    *,
    paths: FinalEvalArtifactPaths,
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    aggregate_records = [
        record for result in matchup_results for record in cast(Sequence[EvalGameRecord], result.get("records", ()))
    ]
    if aggregate_records:
        write_episodes_jsonl(paths.episodes_jsonl, aggregate_records)


def write_canonical_final_eval_sidecars(
    *,
    layout: ArtifactLayout,
    payload: Mapping[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    write_run_level_diagnostics(layout=layout, policy_ids=payload["policy_ids"], matchup_results=matchup_results)
    write_artifact_hashes(layout=layout)


def maybe_layout(output_dir: Path) -> ArtifactLayout | None:
    try:
        return ArtifactLayout.from_final_eval_dir(output_dir)
    except ValueError:
        return None


def write_artifact_hashes(*, layout: ArtifactLayout) -> None:
    tracked_paths = [
        layout.final_eval_summary_json(),
        layout.final_eval_policy_set_json(),
        layout.final_eval_metadata_json(),
        layout.final_eval_matchups_csv(),
        layout.final_eval_posterior_samples_json(),
        layout.final_eval_posterior_samples_npz(),
        layout.final_eval_matrix_csv("mean"),
        layout.seat_bias_json(),
        layout.truncation_heatmap_csv(),
        layout.replay_verification_json(),
        layout.replay_index_json(),
    ]
    payload = {
        "kind": "final_eval_artifact_hashes_v1",
        "artifacts": {layout.relative(path): sha256_file(path) for path in tracked_paths if path.is_file()},
    }
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FINAL_EVAL_ARTIFACT_WRITE_PLAN",
    "FinalEvalArtifactPaths",
    "FinalEvalArtifactWriteStep",
    "build_run_level_seat_bias_payload",
    "final_eval_matchup_manifest_rows",
    "final_eval_artifact_paths",
    "final_eval_artifact_write_plan_payload",
    "legacy_payoff_matrix_name",
    "maybe_layout",
    "relative_to",
    "sha256_file",
    "write_aggregate_final_eval_episodes",
    "write_canonical_final_eval_sidecars",
    "write_core_final_eval_json_artifacts",
    "write_artifact_hashes",
    "write_final_eval_artifacts",
    "write_matrix_csv",
    "write_final_eval_matchup_manifest",
    "write_final_eval_posterior_npz_if_canonical",
    "write_replay_diagnostics",
    "write_run_level_diagnostics",
    "write_truncation_heatmap_csv",
]
