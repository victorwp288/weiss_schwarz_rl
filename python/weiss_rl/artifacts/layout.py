"""Canonical artifact layout helpers for thesis runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Canonical run-relative layout for training, eval, and paper artifacts."""

    run_dir: Path
    training_dir: Path
    training_checkpoints_dir: Path
    training_logs_dir: Path
    training_snapshots_dir: Path
    eval_dir: Path
    final_eval_dir: Path
    final_eval_matchups_dir: Path
    final_eval_matrices_dir: Path
    final_eval_payoff_matrices_dir: Path
    diagnostics_dir: Path
    metagame_dir: Path
    replays_dir: Path
    replays_raw_dir: Path
    replays_bundles_dir: Path
    replays_verification_dir: Path
    tensorboard_dir: Path
    figures_dir: Path
    figures_paper_dir: Path
    manifest_path: Path
    spec_bundle_path: Path
    spec_hash_path: Path
    config_hash_path: Path
    config_json_path: Path
    environment_path: Path
    run_summary_path: Path
    determinism_report_path: Path
    paper_readiness_summary_path: Path
    performance_log_path: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> ArtifactLayout:
        run_dir = Path(run_dir)
        training_dir = run_dir / "training"
        eval_dir = run_dir / "eval"
        final_eval_dir = eval_dir / "final_eval"
        figures_dir = run_dir / "figures"
        return cls(
            run_dir=run_dir,
            training_dir=training_dir,
            training_checkpoints_dir=training_dir / "checkpoints",
            training_logs_dir=training_dir / "logs",
            training_snapshots_dir=training_dir / "snapshots",
            eval_dir=eval_dir,
            final_eval_dir=final_eval_dir,
            final_eval_matchups_dir=final_eval_dir / "matchups",
            final_eval_matrices_dir=final_eval_dir / "matrices",
            final_eval_payoff_matrices_dir=final_eval_dir / "payoff_matrices",
            diagnostics_dir=eval_dir / "diagnostics",
            metagame_dir=eval_dir / "metagame",
            replays_dir=run_dir / "replays",
            replays_raw_dir=(run_dir / "replays" / "raw"),
            replays_bundles_dir=(run_dir / "replays" / "bundles"),
            replays_verification_dir=(run_dir / "replays" / "verification"),
            tensorboard_dir=run_dir / "tensorboard",
            figures_dir=figures_dir,
            figures_paper_dir=figures_dir / "paper",
            manifest_path=run_dir / "manifest.json",
            spec_bundle_path=run_dir / "spec_bundle.json",
            spec_hash_path=run_dir / "spec_hash256.txt",
            config_hash_path=run_dir / "config_hash256.txt",
            config_json_path=run_dir / "config_canonical.json",
            environment_path=run_dir / "environment.json",
            run_summary_path=run_dir / "run_summary.json",
            determinism_report_path=run_dir / "determinism_report.json",
            paper_readiness_summary_path=run_dir / "paper_readiness_summary.json",
            performance_log_path=training_dir / "logs" / "performance.jsonl",
        )

    @classmethod
    def from_final_eval_dir(cls, final_eval_dir: Path) -> ArtifactLayout:
        final_eval_dir = Path(final_eval_dir)
        if final_eval_dir.name != "final_eval" or final_eval_dir.parent.name != "eval":
            raise ValueError("final_eval_dir must resolve to <run_dir>/eval/final_eval for canonical artifact layout")
        return cls.from_run_dir(final_eval_dir.parent.parent)

    def ensure_directories(self) -> None:
        for path in (
            self.run_dir,
            self.training_dir,
            self.training_checkpoints_dir,
            self.training_logs_dir,
            self.training_snapshots_dir,
            self.eval_dir,
            self.final_eval_dir,
            self.final_eval_matchups_dir,
            self.final_eval_matrices_dir,
            self.final_eval_payoff_matrices_dir,
            self.diagnostics_dir,
            self.metagame_dir,
            self.replays_dir,
            self.replays_raw_dir,
            self.replays_bundles_dir,
            self.replays_verification_dir,
            self.tensorboard_dir,
            self.figures_dir,
            self.figures_paper_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        path = Path(path)
        try:
            return path.relative_to(self.run_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def final_eval_matrix_csv(self, field: str) -> Path:
        return self.final_eval_matrices_dir / f"{field}.csv"

    def final_eval_matrix_json(self, field: str) -> Path:
        return self.final_eval_matrices_dir / f"{field}.json"

    def final_eval_payoff_matrix_csv(self, field: str) -> Path:
        return self.final_eval_payoff_matrices_dir / f"{field}.csv"

    def final_eval_payoff_counts_json(self) -> Path:
        return self.final_eval_dir / "payoff_counts.json"

    def final_eval_posterior_samples_json(self) -> Path:
        return self.final_eval_dir / "posterior_samples.json"

    def final_eval_posterior_samples_npz(self) -> Path:
        return self.final_eval_dir / "posterior_samples.npz"

    def final_eval_episodes_jsonl(self) -> Path:
        return self.final_eval_dir / "episodes.jsonl"

    def final_eval_matchups_csv(self) -> Path:
        return self.final_eval_dir / "matchups.csv"

    def final_eval_metadata_json(self) -> Path:
        return self.final_eval_dir / "metadata.json"

    def final_eval_policy_set_json(self) -> Path:
        return self.final_eval_dir / "policy_set.json"

    def final_eval_summary_json(self) -> Path:
        return self.final_eval_dir / "summary.json"

    def final_eval_aggregate_hashes_json(self) -> Path:
        return self.final_eval_dir / "artifact_hashes.json"

    def seat_bias_json(self) -> Path:
        return self.diagnostics_dir / "seat_bias.json"

    def truncation_heatmap_csv(self) -> Path:
        return self.diagnostics_dir / "truncation_heatmap_data.csv"

    def replay_verification_json(self) -> Path:
        return self.diagnostics_dir / "replay_verification.json"

    def replay_index_json(self) -> Path:
        return self.replays_dir / "index.json"


def default_run_dir_name(run_id64: str) -> str:
    return f"run_{run_id64}"
