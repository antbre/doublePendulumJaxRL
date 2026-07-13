"""Checkpointing and policy rollout utilities.

Rather than pickling rejax's dynamically-generated train-state class (which does not
survive pickling), we save only what ``PPO.make_act`` actually needs: the actor
network parameters and the observation-normalisation statistics. On load we rebuild
the algorithm from its config and hand ``make_act`` a lightweight stand-in state.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .config import ActuationMode, EnvParams
from .env import DoublePendulum
from .train import build_algo


# ---------------------------------------------------------------------------- #
# checkpointing
# ---------------------------------------------------------------------------- #
def save_checkpoint(path: str, mode: ActuationMode, config: Dict[str, Any], train_state) -> None:
    """Persist everything needed to reconstruct the greedy policy."""
    payload = {
        "mode": int(ActuationMode(mode)),
        "config": config,
        "actor_params": jax.device_get(train_state.actor_ts.params),
        "normalize_observations": bool(config.get("normalize_observations", False)),
    }
    if payload["normalize_observations"]:
        rms = train_state.rms_state
        payload["obs_rms"] = {
            "mean": jax.device_get(rms.mean),
            "var": jax.device_get(rms.var),
            "count": jax.device_get(rms.count),
        }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_policy(path: str):
    """Load a checkpoint and return ``(act_fn, env, env_params)``.

    ``act_fn`` has signature ``act(obs, rng) -> action`` (matching rejax).
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    mode = ActuationMode(payload["mode"])
    # Rebuild the algorithm so we get the correct actor module back.
    algo, _ = build_algo(mode, **payload["config"])

    fake_ts = SimpleNamespace(
        actor_ts=SimpleNamespace(params=payload["actor_params"]),
    )
    if payload["normalize_observations"]:
        rms = payload["obs_rms"]
        fake_ts.rms_state = SimpleNamespace(
            mean=jnp.asarray(rms["mean"]),
            var=jnp.asarray(rms["var"]),
            count=jnp.asarray(rms["count"]),
        )

    act = algo.make_act(fake_ts)
    return act, algo.env, algo.env_params


# ---------------------------------------------------------------------------- #
# rollout
# ---------------------------------------------------------------------------- #
def rollout(act, env: DoublePendulum, env_params: EnvParams, seed: int = 0):
    """Roll out one episode and return a dict of NumPy trajectory arrays."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, env_params)

    n_steps = int(env_params.max_steps_in_episode)

    def step(carry, _):
        obs, state, key = carry
        key, act_key, step_key = jax.random.split(key, 3)
        action = act(obs, act_key)
        next_obs, next_state, reward, done, info = env.step(
            step_key, state, action, env_params
        )
        out = {
            "theta1": state.theta1,
            "theta2": state.theta2,
            "omega1": state.omega1,
            "omega2": state.omega2,
            "action": jnp.atleast_1d(action),
            "reward": reward,
            "tip_height": info["tip_height"],
        }
        return (next_obs, next_state, key), out

    _, traj = jax.lax.scan(step, (obs, state, key), None, length=n_steps)
    return jax.tree.map(lambda x: np.asarray(x), traj)
