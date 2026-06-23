from __future__ import annotations

from weiss_rl.runtime.components.action_catalog_setup import resolve_runtime_action_catalog_setup


def test_runtime_action_catalog_setup_defaults_without_spec_bundle() -> None:
    setup = resolve_runtime_action_catalog_setup(spec_bundle=None)

    assert setup.action_catalog is None
    assert setup.action_family_index == {}
    assert setup.action_attack_type_index == {}
    assert setup.last_action_arg0_obs_index == -1


def test_runtime_action_catalog_setup_finds_last_action_arg0_header_index() -> None:
    setup = resolve_runtime_action_catalog_setup(
        spec_bundle={
            "observation": {
                "header_fields": [
                    {"name": "turn_player", "index": 0},
                    {"name": "last_action_arg0", "index": "7"},
                ]
            }
        }
    )

    assert setup.last_action_arg0_obs_index == 7


def test_runtime_action_catalog_setup_ignores_missing_or_malformed_observation_spec() -> None:
    assert resolve_runtime_action_catalog_setup(spec_bundle={"observation": []}).last_action_arg0_obs_index == -1
    assert (
        resolve_runtime_action_catalog_setup(
            spec_bundle={"observation": {"header_fields": []}}
        ).last_action_arg0_obs_index
        == -1
    )


def test_runtime_action_catalog_setup_tolerates_non_action_catalog_spec() -> None:
    setup = resolve_runtime_action_catalog_setup(spec_bundle={"observation": {"header_fields": []}})

    assert setup.action_catalog is None
    assert setup.action_family_index == {}
    assert setup.action_attack_type_index == {}
