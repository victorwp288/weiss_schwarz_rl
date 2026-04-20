from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from weiss_rl.artifact_hygiene import run_artifact_hygiene_scan, scan_artifact_roots, scan_tracked_repo_tree
from weiss_rl.toy_public_demo import stage_public_demo_run

REPO_ROOT = Path(__file__).resolve().parents[3]


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Victor"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "victor@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_demo_artifacts_pass_hygiene_scan(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "toy_public_demo"
    stage_public_demo_run(run_dir)

    summary = run_artifact_hygiene_scan(repo_root=REPO_ROOT, artifact_roots=(run_dir,))

    assert summary.findings == ()
    assert summary.artifact_file_count > 0


def test_scan_artifact_roots_flags_trademark_marker_in_text_like_file(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runs" / "bad"
    artifact_root.mkdir(parents=True)
    payload = {"bundle_kind": "public_data", "series": "Weiss Schwarz"}
    (artifact_root / "catalog.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    findings, _stats = scan_artifact_roots((artifact_root,))

    assert any(finding.rule == "trademark_marker" for finding in findings)


def test_scan_artifact_roots_flags_card_text_and_suspicious_image_assets(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runs" / "bad"
    logos_dir = artifact_root / "logos"
    logos_dir.mkdir(parents=True)
    (artifact_root / "card_texts.csv").write_text("card_text,id\nWhen this card attacks,1\n", encoding="utf-8")
    (logos_dir / "brandmark.png").write_bytes(b"png")

    findings, _stats = scan_artifact_roots((artifact_root,))
    finding_rules = {finding.rule for finding in findings}

    assert "card_text_field" in finding_rules
    assert "suspicious_image_asset" in finding_rules
    assert "suspicious_text_asset" in finding_rules


def test_scan_artifact_roots_scans_replay_bundle_members(tmp_path: Path) -> None:
    replay_dir = tmp_path / "runs" / "scan" / "replays"
    replay_dir.mkdir(parents=True)
    replay_path = replay_dir / "replay_deadbeef00000000.zip"
    with zipfile.ZipFile(replay_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("meta.json", json.dumps({"franchise_name": "Marvel"}) + "\n")
        archive.writestr("steps.jsonl", json.dumps({"t": 0, "action": 1}) + "\n")

    findings, _stats = scan_artifact_roots((tmp_path / "runs",))

    assert any(finding.path.endswith("::meta.json") for finding in findings)
    assert any(finding.rule == "trademark_marker" for finding in findings)


def test_scan_artifact_roots_matches_path_markers_on_tokens_only(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runs" / "bad"
    artifact_root.mkdir(parents=True)
    (artifact_root / "icon.png").write_bytes(b"png")
    (artifact_root / "lexicon.png").write_bytes(b"png")

    findings, _stats = scan_artifact_roots((artifact_root,))
    flagged_names = {Path(finding.path).name for finding in findings if finding.rule == "suspicious_image_asset"}

    assert "icon.png" in flagged_names
    assert "lexicon.png" not in flagged_names


def test_scan_tracked_repo_tree_skips_out_of_scope_doc_assets_but_scans_tracked_data(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "bundles"
    docs_dir.mkdir()
    data_dir.mkdir()
    (docs_dir / "README.txt").write_text("Weiss Schwarz reference text\n", encoding="utf-8")
    (docs_dir / "logo.png").write_bytes(b"png")
    (data_dir / "catalog.json").write_text(json.dumps({"series": "Weiss Schwarz"}) + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "docs/README.txt", "docs/logo.png", "bundles/catalog.json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    findings, _stats = scan_tracked_repo_tree(tmp_path)
    finding_paths = {finding.path for finding in findings}

    assert "bundles/catalog.json" in finding_paths
    assert "docs/README.txt" not in finding_paths
    assert "docs/logo.png" not in finding_paths


def test_artifact_scan_entrypoint_returns_nonzero_on_findings(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("scratch repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    artifact_root = tmp_path / "runs" / "scan"
    artifact_root.mkdir(parents=True)
    (artifact_root / "catalog.json").write_text(json.dumps({"series": "Weiss Schwarz"}) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "artifact_scan.py"),
            "--repo-root",
            str(tmp_path),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Artifact hygiene scan failed" in result.stderr
    assert "trademark_marker" in result.stderr
