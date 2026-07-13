"""Training entry point: rejax PPO on the double-pendulum environment.

The environment is fully jitted, so ``rejax`` runs all ``num_envs`` copies and the
whole PPO update on-device (GPU if one is present, otherwise CPU).
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from rejax import PPO
from rejax.evaluate import evaluate as rejax_evaluate

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
        ent_coef=0.01,  # nonzero entropy bonus so PPO keeps exploring (swing-up)
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


def _install_eval_callback(algo: PPO, verbose: bool) -> PPO:
    """Replace rejax's eval callback with a bottom-start evaluation (+ live logging).

    Training may use Reference-State Initialization (``p_start_top > 0``), which would
    otherwise inflate/obscure the eval return with easy top-starts. We evaluate with
    ``p_start_top = 0`` so the reported return is an honest "swing up from hanging"
    measure. The whole train loop is jitted, so we log via a host-side callback that
    still streams live.
    """
    env = algo.env
    eval_params = algo.env_params.replace(p_start_top=0.0)
    max_steps = algo.env_params.max_steps_in_episode

    def _log(step, ret, std, ln):  # runs on the host with concrete numpy values
        print(
            f"step={int(step):>10}  return={float(ret):8.2f} +/- {float(std):6.2f}"
            f"  ep_len={float(ln):6.1f}",
            flush=True,
        )

    def eval_callback(a: PPO, ts, rng):
        act = a.make_act(ts)
        lengths, returns = rejax_evaluate(act, rng, env, eval_params, 128, max_steps)
        if verbose:
            jax.debug.callback(
                _log, ts.global_step, returns.mean(), returns.std(), lengths.mean()
            )
        return lengths, returns

    return algo.replace(eval_callback=eval_callback)


def _concat_eval(a, b):
    """Concatenate two rejax evaluation pytrees along the eval-point axis."""
    if a is None:
        return b
    if b is None:
        return a
    return jax.tree.map(lambda x, y: jnp.concatenate([jnp.asarray(x), jnp.asarray(y)]), a, b)


def train(
    mode: ActuationMode,
    seed: int = 0,
    verbose: bool = True,
    checkpoint_every: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    **overrides: Any,
):
    """Train a policy. Returns ``(algo, train_state, evaluation, config)``.

    ``evaluation`` is ``(lengths, returns)``, each of shape
    ``(num_eval_points, num_eval_seeds)`` — average over the last axis for a curve.

    With ``verbose=True`` a progress line is streamed at every ``eval_freq`` steps.

    Checkpointing / resuming
    ------------------------
    * ``resume_from``: path to a checkpoint to continue training from. ``mode`` and the
      saved ``config`` are taken from the checkpoint; ``overrides`` (e.g. a new
      ``total_timesteps`` = *additional* steps) still apply, and ``mode`` is ignored.
    * ``checkpoint_every``: if set, training runs in Python-level chunks of this many
      env steps and a full (resumable) checkpoint is written after each chunk to
      ``<checkpoint_path stem>_step<STEP>.pkl``. Requires ``checkpoint_path``.
    """
    # lazy import to avoid a circular import (evaluate imports build_algo from here)
    from .evaluate import save_checkpoint, load_train_state

    resume_state = None
    prev_eval = None
    if resume_from is not None:
        mode, config, resume_state, prev_eval = load_train_state(resume_from)
        config = dict(config)
        config.update(overrides)
        algo, config = build_algo(mode, **config)
        if verbose:
            print(f"Resuming from {resume_from} at step {int(resume_state.global_step)}")
    else:
        algo, config = build_algo(mode, **overrides)

    algo = _install_eval_callback(algo, verbose=verbose)
    rng = jax.random.PRNGKey(seed)

    # --- single-shot training (no periodic checkpoints) ---
    if checkpoint_every is None:
        train_fn = jax.jit(algo.train)
        if resume_state is None:
            train_state, evaluation = train_fn(rng)
        else:
            train_state, evaluation = train_fn(rng, resume_state)
        return algo, train_state, _concat_eval(prev_eval, evaluation), config

    # --- chunked training with a full checkpoint after each chunk ---
    if checkpoint_path is None:
        raise ValueError("checkpoint_every requires checkpoint_path")
    stem, _ext = os.path.splitext(checkpoint_path)

    total = int(config["total_timesteps"])
    seg = int(checkpoint_every)
    n_chunks = max(1, math.ceil(total / seg))
    # Each chunk trains `seg` steps; skip the per-chunk initial eval to avoid duplicates.
    seg_algo = algo.replace(total_timesteps=seg, skip_initial_evaluation=resume_state is not None)
    seg_train = jax.jit(seg_algo.train)

    train_state = resume_state
    evaluation = prev_eval
    for i in range(n_chunks):
        if train_state is None:
            train_state, ev = seg_train(rng)
        else:
            train_state, ev = seg_train(rng, train_state)
        evaluation = _concat_eval(evaluation, ev)
        step = int(train_state.global_step)
        path_i = f"{stem}_step{step}{_ext or '.pkl'}"
        save_checkpoint(path_i, mode, config, train_state, evaluation=evaluation)
        if verbose:
            print(f"  [checkpoint] {i + 1}/{n_chunks} -> {path_i}", flush=True)

    return algo, train_state, evaluation, config
