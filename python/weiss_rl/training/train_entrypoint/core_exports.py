"""Core public workflow exports for the training entrypoint facade."""

from __future__ import annotations

# ruff: noqa: F401
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from weiss_rl.artifacts.manifest import (
    RunArtifacts,
    RunManifest,
    build_seed_file_manifest,
    write_run_artifacts,
)
from weiss_rl.config import (
    StackConfig,
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)
from weiss_rl.core.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.core.spec import assert_spec_bundle_contract
from weiss_rl.diagnostics.cli_banner import print_startup_banner
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.experiments.toy_public_demo import (
    PUBLIC_DEMO_MODE,
    public_demo_simulator_info,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    stage_public_demo_run,
)
from weiss_rl.league import run_promotion_gate as run_promotion_gate
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.model import PolicyValueModel
from weiss_rl.model import build_policy_value_model as build_policy_value_model
from weiss_rl.models.loading import load_snapshot_eval_model
from weiss_rl.runtime import QueueRuntime, QueueRuntimeMode
from weiss_rl.runtime import build_runtime_config as build_runtime_config
from weiss_rl.training.cli import build_train_parser
from weiss_rl.training.execution import resolve_training_execution_settings
from weiss_rl.training.report_payloads import (
    augment_determinism_payload,
    augment_environment_payload,
    augment_run_summary_payload,
    profiling_enabled_message,
)
from weiss_rl.training.run_identity import new_run_identity, resume_run_identity
from weiss_rl.training.startup import (
    apply_training_flag_overrides,
    manifest_scaffold_only_reason,
    noleague_training_prerequisite_failure,
    print_manifest_only_message,
    raise_noleague_training_prerequisite_failure,
    raise_runtime_prerequisite_failure,
    resolve_device,
    resolve_runtime_profile,
    resolve_seed,
    runtime_training_prerequisite_failure,
)
from weiss_rl.training.train_entrypoint.cli import (
    TrainCliState,
    TrainManifestState,
    TrainStartupState,
    execute_train_run,
    prepare_train_manifest_state,
    prepare_train_startup_state,
    resolve_train_cli_state,
)

_CORE_EXPORT_NAMES = (
    "Mapping",
    "Sequence",
    "Path",
    "Any",
    "cast",
    "torch",
    "nn",
    "RunArtifacts",
    "RunManifest",
    "build_seed_file_manifest",
    "write_run_artifacts",
    "StackConfig",
    "apply_stack_overrides",
    "canonical_config_dict",
    "compute_config_hash256",
    "load_stack_config",
    "parse_override_tokens",
    "SimulatorContract",
    "load_verified_simulator_contract",
    "assert_spec_bundle_contract",
    "print_startup_banner",
    "TensorBoardLogger",
    "tensorboard_unavailable_reason",
    "PUBLIC_DEMO_MODE",
    "public_demo_simulator_info",
    "public_demo_spec_bundle",
    "public_demo_spec_hash256",
    "stage_public_demo_run",
    "run_promotion_gate",
    "ImpalaLearner",
    "PpoLiteLearner",
    "PolicyValueModel",
    "build_policy_value_model",
    "load_snapshot_eval_model",
    "QueueRuntime",
    "QueueRuntimeMode",
    "build_runtime_config",
    "build_train_parser",
    "resolve_training_execution_settings",
    "augment_determinism_payload",
    "augment_environment_payload",
    "augment_run_summary_payload",
    "profiling_enabled_message",
    "new_run_identity",
    "resume_run_identity",
    "apply_training_flag_overrides",
    "manifest_scaffold_only_reason",
    "noleague_training_prerequisite_failure",
    "print_manifest_only_message",
    "raise_noleague_training_prerequisite_failure",
    "raise_runtime_prerequisite_failure",
    "resolve_device",
    "resolve_runtime_profile",
    "resolve_seed",
    "runtime_training_prerequisite_failure",
    "TrainCliState",
    "TrainManifestState",
    "TrainStartupState",
    "execute_train_run",
    "prepare_train_manifest_state",
    "prepare_train_startup_state",
    "resolve_train_cli_state",
)

CORE_COMPAT_EXPORTS: Mapping[str, Any] = {name: globals()[name] for name in _CORE_EXPORT_NAMES}
