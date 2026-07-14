"""Checkpointing and policy rollout utilities.

A checkpoint stores two things:

* the actor parameters + observation-normalisation stats that ``PPO.make_act`` needs
  to *evaluate* the policy (used by :func:`load_policy`), and
* the **full** rejax train state serialised with ``flax.serialization`` (optimizer
  state, critic, RNG, global step, ...), which is what lets training **resume** from
  the checkpoint (used by :func:`load_train_state`).

rejax's train state is a dynamically-generated class that does not survive pickling,
so we serialise it to bytes with flax and, on load, restore it into a fresh template
produced by ``algo.init_state``.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .config import ActuationMode, EnvParams
from .env import DoublePendulum
from .train import build_algo


# ---------------------------------------------------------------------------- #
# checkpointing
# ---------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    mode: ActuationMode,
    config: Dict[str, Any],
    train_state,
    evaluation: Optional[Any] = None,
) -> None:
    """Persist the policy *and* the full train state (so training can resume)."""
    payload = {
        "mode": int(ActuationMode(mode)),
        "config": config,
        "actor_params": jax.device_get(train_state.actor_ts.params),
        "normalize_observations": bool(config.get("normalize_observations", False)),
        # Full train state for resuming (optimizer/critic/rng/step/...).
        "train_state": serialization.to_bytes(train_state),
        "global_step": int(train_state.global_step),
    }
    if payload["normalize_observations"]:
        # rejax >=0.1 uses obs_rms_state; older versions used rms_state.
        rms = getattr(train_state, "obs_rms_state", None) or getattr(
            train_state, "rms_state", None
        )
        if rms is None:
            raise AttributeError(
                "train_state has normalize_observations=True but no obs RMS state"
            )
        payload["obs_rms"] = {
            "mean": jax.device_get(rms.mean),
            "var": jax.device_get(rms.var),
            "count": jax.device_get(rms.count),
        }
    if evaluation is not None:
        payload["evaluation"] = jax.device_get(evaluation)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_train_state(path: str):
    """Load a checkpoint's full train state for resuming.

    Returns ``(mode, config, train_state, evaluation)`` where ``evaluation`` is the
    accumulated learning-curve history saved so far (or ``None``).
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if "train_state" not in payload:
        raise ValueError(
            f"Checkpoint {path!r} has no full train state and cannot be resumed "
            "(it was saved by an older version). Retrain to produce a resumable one."
        )
    mode = ActuationMode(payload["mode"])
    algo, config = build_algo(mode, **payload["config"])
    template = algo.init_state(jax.random.PRNGKey(0))
    train_state = serialization.from_bytes(template, payload["train_state"])
    return mode, config, train_state, payload.get("evaluation")


def load_policy(path: str, deterministic: bool = True):
    """Load a checkpoint and return ``(act_fn, env, env_params)``.

    ``act_fn`` has signature ``act(obs, rng) -> action`` (matching rejax).

    With ``deterministic=True`` (default) the policy returns the Gaussian *mean*
    action instead of a sample. rejax's training-time ``act`` samples from a policy
    whose std is ~1, which injects large per-step noise and looks like bang-bang
    chatter; the mean is the actual learned controller and is what you want to
    evaluate/visualise.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    mode = ActuationMode(payload["mode"])
    # Rebuild the algorithm so we get the correct actor module back.
    algo, _ = build_algo(mode, **payload["config"])
    params = payload["actor_params"]
    normalize = payload["normalize_observations"]

    if normalize:
        rms = payload["obs_rms"]
        mean = jnp.asarray(rms["mean"])
        std = jnp.sqrt(jnp.asarray(rms["var"]) + 1e-8)

    if not deterministic:
        fake_ts = SimpleNamespace(actor_ts=SimpleNamespace(params=params))
        if normalize:
            fake_ts.obs_rms_state = SimpleNamespace(
                mean=mean, var=jnp.asarray(rms["var"]), count=jnp.asarray(rms["count"])
            )
        return algo.make_act(fake_ts), algo.env, algo.env_params

    actor = algo.actor
    low, high = actor.action_range

    def act(obs, rng):
        if normalize:
            obs = (obs - mean) / std
        obs = jnp.expand_dims(obs, 0)
        dist = actor.apply(params, obs, method="_action_dist")
        action = jnp.clip(dist.mode(), low, high)  # mean action, no sampling noise
        return jnp.squeeze(action)

    return act, algo.env, algo.env_params


# ---------------------------------------------------------------------------- #
# rollout
# ---------------------------------------------------------------------------- #
def rollout(act, env: DoublePendulum, env_params: EnvParams, seed: int = 0):
    """Roll out one episode and return a dict of NumPy trajectory arrays.

    Always starts from the bottom (``p_start_top = 0``) so the animation shows the full
    swing-up, regardless of any RSI curriculum used during training.
    """
    env_params = env_params.replace(p_start_top=0.0)
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
            "torque": info["torque"],  # [shoulder, elbow] applied torque
        }
        return (next_obs, next_state, key), out

    _, traj = jax.lax.scan(step, (obs, state, key), None, length=n_steps)
    return jax.tree.map(lambda x: np.asarray(x), traj)
