from __future__ import annotations

from typing import Any, Literal

Profile = Literal["debug", "balanced", "fast"]
LayoutName = Literal["mask", "i16_legal_ids"]

"""Build weiss_sim EnvPool from env_config and select an RL layout label by profile.

    Returns:
        (pool, layout_name)

    Profile mapping (master plan §5.3 + task):
      - debug -> legal_repr=mask_u8, obs_dtype=i32, layout_name="mask"
      - fast  -> legal_repr=ids_u16, obs_dtype=i16, layout_name="i16_legal_ids"
      - balanced -> legal_repr=mask_u8, obs_dtype=i16, layout_name="mask"

    Note:
      This factory calls weiss_sim.fast/inspect to obtain a WeissEnv and returns env.pool.
 """

def make_env_pool_from_config(
    env_config: dict[str, Any],
    *,
    profile: Profile,
) -> tuple[Any, LayoutName]:
    if profile == "debug":
        legal_repr: str = "mask_u8"
        obs_dtype: str = "i32"
        layout_name: LayoutName = "mask"
        entrypoint = "inspect"
    elif profile == "fast":
        legal_repr = "ids_u16"
        obs_dtype = "i16"
        layout_name = "i16_legal_ids"
        entrypoint = "fast"
    else:  # balanced
        legal_repr = "mask_u8"
        obs_dtype = "i16"
        layout_name = "mask"
        entrypoint = "fast"

    # Import locally so weiss_rl can be imported without weiss_sim installed.
    import weiss_sim  # type: ignore

    kwargs = dict(env_config)
    kwargs["legal_repr"] = legal_repr
    kwargs["obs_dtype"] = obs_dtype

    # Optional: fail early if caller forgot key surfaces.
    required = ("num_envs", "max_decisions", "max_ticks", "observation_visibility")
    missing = [k for k in required if k not in kwargs]
    if missing:
        raise ValueError(f"env_config missing required keys: {missing}")

    env = weiss_sim.inspect(**kwargs) if entrypoint == "inspect" else weiss_sim.fast(**kwargs)
    return env.pool, layout_name
