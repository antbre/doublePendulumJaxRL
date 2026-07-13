"""Training entry point: rejax PPO on the double-pendulum environment.

The environment is fully jitted, so ``rejax`` runs all ``num_envs`` copies and the
whole PPO update on-device (GPU if one is present, otherwise CPU).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import jax
from rejax import PPO

from .config import ActuationMode
from .env import DoublePendulum


def default_ppo_config(mode: ActuationMode) -> Dict[str, Any]:
    """Reasonable PPO defaults for the double pendulum.

    The underactuated modes (top-only / bottom-only) are much harder swing-up
    problems, so they get a bigger timestep budget by default.
    """
    config: Dict[str, Any] = dict(
        total_timesteps=1_000_000,
        eval_freq=25_000,
        num_envs=128,
        num_steps=128,
        num_epochs=8,
        num_minibatches=16,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        normalize_observations=True,
        agent_kwargs=dict(activation="tanh", hidden_layer_sizes=(128, 128)),
    )
    if ActuationMode(mode) != ActuationMode.BOTH:
        config["total_timesteps"] = 3_000_000
    return config


def build_algo(mode: ActuationMode, **overrides: Any) -> Tuple[PPO, Dict[str, Any]]:
    """Construct a configured (but untrained) PPO algorithm for ``mode``."""
    mode = ActuationMode(mode)
    env = DoublePendulum(mode)
    config = default_ppo_config(mode)
    config.update(overrides)
    # rejax mutates the (nested) config in place — e.g. it replaces the "tanh"
    # activation string with the nn.tanh function, which is not picklable. Give it a
    # deep copy so the config we return (and later checkpoint) stays clean/serialisable.
    algo = PPO.create(env=env, env_params=env.default_params, **copy.deepcopy(config))
    return algo, config


def _with_progress_logging(algo: PPO) -> PPO:
    """Wrap rejax's eval callback so it streams a progress line at every eval point.

    The whole training loop is jitted, so a normal ``print`` never runs mid-training;
    ``jax.debug.print`` fires from inside the compiled loop and streams live.
    """
    base_callback = algo.eval_callback

    def _log(step, ret, std, ln):  # runs on the host with concrete numpy values
        print(
            f"step={int(step):>10}  return={float(ret):8.2f} +/- {float(std):6.2f}"
            f"  ep_len={float(ln):6.1f}",
            flush=True,
        )

    def logging_callback(a: PPO, ts, rng):
        lengths, returns = base_callback(a, ts, rng)
        # Host-side callback (not jax.debug.print) so we get proper float formatting;
        # it streams live even though the whole train loop is jitted.
        jax.debug.callback(
            _log, ts.global_step, returns.mean(), returns.std(), lengths.mean()
        )
        return lengths, returns

    return algo.replace(eval_callback=logging_callback)


def train(mode: ActuationMode, seed: int = 0, verbose: bool = True, **overrides: Any):
    """Train a policy. Returns ``(algo, train_state, evaluation, config)``.

    ``evaluation`` is ``(lengths, returns)``, each of shape
    ``(num_eval_points, num_eval_seeds)`` — average over the last axis for a curve.

    With ``verbose=True`` a progress line is streamed at every ``eval_freq`` steps.
    """
    algo, config = build_algo(mode, **overrides)
    if verbose:
        algo = _with_progress_logging(algo)
    train_fn = jax.jit(algo.train)
    train_state, evaluation = train_fn(jax.random.PRNGKey(seed))
    return algo, train_state, evaluation, config
