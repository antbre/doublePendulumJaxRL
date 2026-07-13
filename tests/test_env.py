"""gymnax-API contract tests for the DoublePendulum environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from double_pendulum_jaxrl import ActuationMode, DoublePendulum
from double_pendulum_jaxrl.env import EnvState


@pytest.mark.parametrize(
    "mode,expected_dim",
    [(ActuationMode.BOTH, 2), (ActuationMode.TOP_ONLY, 1), (ActuationMode.BOTTOM_ONLY, 1)],
)
def test_action_dim(mode, expected_dim):
    env = DoublePendulum(mode)
    assert env.action_space().shape == (expected_dim,)
    assert env.num_actions == expected_dim


def test_reset_step_shapes():
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert obs.shape == (6 + env.act_dim,)  # 6 state features + previous action
    action = jnp.zeros(2)
    obs2, state2, reward, done, info = env.step(jax.random.PRNGKey(1), state, action, params)
    assert obs2.shape == (6 + env.act_dim,)
    assert reward.shape == ()
    assert done.dtype == jnp.bool_
    assert state2.time == 1


def test_jit_and_vmap_over_keys():
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), 32)

    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    obs, state = reset(keys, params)
    assert obs.shape == (32, 6 + env.act_dim)

    actions = jnp.zeros((32, 2))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    obs2, state2, reward, done, info = step(keys, state, actions, params)
    assert obs2.shape == (32, 6 + env.act_dim)
    assert reward.shape == (32,)


def test_determinism():
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    key = jax.random.PRNGKey(42)
    o1, s1 = env.reset(key, params)
    o2, s2 = env.reset(key, params)
    assert np.allclose(np.asarray(o1), np.asarray(o2))


def test_time_limit_termination():
    env = DoublePendulum(ActuationMode.BOTTOM_ONLY)
    params = env.default_params.replace(max_steps_in_episode=3)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    done = False
    for _ in range(3):
        _, state, _, done, _ = env.step(key, state, jnp.zeros(1), params)
    assert bool(done)


def test_torque_routing():
    """Top-only / bottom-only must not drive the other joint."""
    params = DoublePendulum(ActuationMode.BOTH).default_params
    top = DoublePendulum(ActuationMode.TOP_ONLY)
    bottom = DoublePendulum(ActuationMode.BOTTOM_ONLY)
    tau_top = top._action_to_torque(jnp.array([1.0]), params)
    tau_bottom = bottom._action_to_torque(jnp.array([1.0]), params)
    assert float(tau_top[1]) == 0.0 and float(tau_top[0]) != 0.0
    assert float(tau_bottom[0]) == 0.0 and float(tau_bottom[1]) != 0.0


def test_gated_reward_and_action_smoothness():
    """Swing-up height is rewarded when low; action jumps are penalised everywhere."""
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    hanging = EnvState(
        theta1=0.0,
        theta2=0.0,
        omega1=0.0,
        omega2=0.0,
        last_action=jnp.zeros(2),
        time=0,
    )
    upright = EnvState(
        theta1=jnp.pi,
        theta2=0.0,
        omega1=0.0,
        omega2=0.0,
        last_action=jnp.zeros(2),
        time=0,
    )
    hanging_r = env._reward(hanging, jnp.zeros(2), jnp.zeros(2), params)
    upright_r = env._reward(upright, jnp.zeros(2), jnp.zeros(2), params)
    smooth_r = env._reward(hanging, jnp.zeros(2), jnp.zeros(2), params)
    jerky_r = env._reward(hanging, jnp.ones(2), jnp.zeros(2), params)
    assert float(upright_r) > float(hanging_r)
    assert float(jerky_r) < float(smooth_r)
