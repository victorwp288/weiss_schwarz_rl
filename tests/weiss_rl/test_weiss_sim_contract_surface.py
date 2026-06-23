from __future__ import annotations

from .rl_step_layout_contract_test_support import MIN_WEISS_SIM_VERSION, sim_module, version_tuple


def test_weiss_sim_12_contract_surface_is_available() -> None:
    sim = sim_module()
    version = getattr(sim, "__version__", "")

    assert version_tuple(version) >= MIN_WEISS_SIM_VERSION
    assert int(sim.OBS_LEN) == 378
    assert int(sim.ACTION_SPACE_SIZE) == 527
    assert int(sim.SPEC_HASH) == 8590000130
    assert hasattr(sim, "make_pool")
    assert hasattr(sim, "EnvPoolBuffers")
    assert hasattr(sim, "export_spec_bundle")
    assert hasattr(sim.EnvPoolBuffers, "step_sample_from_logits_with_logp")
    assert hasattr(sim.EnvPoolBuffers, "legal_action_context_v1")

    bundle = sim.export_spec_bundle()
    assert int(bundle["observation"]["obs_len"]) == int(sim.OBS_LEN)
    assert int(bundle["action"]["action_space_size"]) == int(sim.ACTION_SPACE_SIZE)
    assert int(bundle["spec_hash"]) == int(sim.SPEC_HASH)
