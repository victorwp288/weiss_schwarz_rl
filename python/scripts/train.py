from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import load_stack_config
from weiss_rl.spec import RuntimeSpecBundle, verify_runtime_spec_bundle


def _default_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_runtime_spec_artifacts(run_dir: Path, runtime_spec: RuntimeSpecBundle) -> None:
    (run_dir / "spec_bundle.json").write_text(
        json.dumps(runtime_spec.bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "spec_hash256.txt").write_text(runtime_spec.bundle_hash + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scaffold entrypoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--spec-hash", type=str, default="", help="Spec hash for contract validation")
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument("--run-id", type=str, default="", help="Run identifier for reproducibility")
    args = parser.parse_args()

    run_id = args.run_id.strip() or _default_run_id()
    stack = load_stack_config(args.stack_config)
    runtime_spec = verify_runtime_spec_bundle(
        args.spec_hash,
        require_export_spec_bundle=stack.require_export_spec_bundle,
        persist_in_manifest=stack.persist_spec_bundle_in_manifest,
    )

    print_startup_banner(
        args.spec_hash,
        args.config_hash,
        run_id,
        spec_mismatch_policy=stack.spec_mismatch_policy,
    )
    if runtime_spec is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={runtime_spec.spec_hash} sha256={runtime_spec.bundle_hash}"
        )

    print(f"Loaded stack config with {len(stack.components)} components")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if runtime_spec is not None and stack.persist_spec_bundle_in_manifest:
        _write_runtime_spec_artifacts(run_dir, runtime_spec)
    if args.config_hash.strip():
        (run_dir / "config_hash256.txt").write_text(args.config_hash.strip() + "\n", encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stack_config": str(args.stack_config),
        "spec_hash": args.spec_hash,
        "config_hash": args.config_hash,
        "spec_mismatch_policy": stack.spec_mismatch_policy,
        "require_export_spec_bundle": stack.require_export_spec_bundle,
        "persist_spec_bundle_in_manifest": stack.persist_spec_bundle_in_manifest,
        "observed_spec_hash": runtime_spec.spec_hash if runtime_spec is not None else "",
        "observed_spec_bundle_hash": runtime_spec.bundle_hash if runtime_spec is not None else "",
        "component_count": len(stack.components),
        "seed_set_count": len(stack.seed_sets),
        "note": "Smoke run: config loading only (no training executed).",
    }
    if runtime_spec is not None and stack.persist_spec_bundle_in_manifest:
        manifest["spec_bundle"] = runtime_spec.bundle
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
