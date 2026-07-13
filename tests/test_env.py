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


def _state(theta1, theta2=0.0, omega1=0.0, omega2=0.0):
    return EnvState(
        theta1=jnp.asarray(theta1),
        theta2=jnp.asarray(theta2),
        omega1=jnp.asarray(omega1),
        omega2=jnp.asarray(omega2),
        last_action=jnp.zeros(2),
        time=0,
    )


def _r(env, params, state, action, prev=None):
    """Evaluate the reward, defaulting prev_action = action (zero action-rate penalty)."""
    prev = action if prev is None else prev
    return float(env._reward(state, action, prev, params))


def test_reward_upright_and_actuation():
    """Upright is rewarded over hanging; actuation is penalised quadratically."""
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    hanging_r = _r(env, params, _state(0.0), jnp.zeros(2))
    upright_r = _r(env, params, _state(jnp.pi), jnp.zeros(2))
    # Upright (quadratic angle-error reward is 0 at the target) beats hanging.
    assert upright_r > hanging_r
    # Actuation costs: driving the joints is worse than doing nothing at the same state.
    idle_r = _r(env, params, _state(jnp.pi), jnp.zeros(2))
    active_r = _r(env, params, _state(jnp.pi), jnp.ones(2))
    assert active_r < idle_r


def test_reward_dense_height_below_horizontal():
    """The dense height term rewards progress even below horizontal (guides the swing-up).

    Both states are below the horizontal line, so the max(tip_height, 0) swing bonus is
    zero for each; the higher one must still score better thanks to the dense w_height term.
    """
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    low_r = _r(env, params, _state(0.1), jnp.zeros(2))   # tip_height ~ -1 (near bottom)
    high_r = _r(env, params, _state(1.2), jnp.zeros(2))  # tip_height ~ -0.36, still < 0
    assert high_r > low_r


def test_reward_swing_bonus_above_horizontal():
    """The swing bonus only rewards the tip once it rises above the horizontal line."""
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    # tip_height < 0 (below horizontal) gets no swing bonus; tip_height > 0 does.
    below_r = _r(env, params, _state(0.3), jnp.zeros(2))
    above_r = _r(env, params, _state(jnp.pi - 0.3), jnp.zeros(2))
    assert above_r > below_r


def test_reward_velocity_penalty_gated_near_target():
    """Joint velocity is penalised near the target but essentially free far from it."""
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    # Near the target: adding velocity should noticeably reduce the reward.
    near_still = _r(env, params, _state(jnp.pi), jnp.zeros(2))
    near_fast = _r(env, params, _state(jnp.pi, omega1=3.0, omega2=3.0), jnp.zeros(2))
    near_drop = near_still - near_fast
    # Far from the target (hanging): the same velocity is barely penalised (gate ~0).
    far_still = _r(env, params, _state(0.0), jnp.zeros(2))
    far_fast = _r(env, params, _state(0.0, omega1=3.0, omega2=3.0), jnp.zeros(2))
    far_drop = far_still - far_fast
    assert near_drop > 0.0
    assert near_drop > far_drop


def test_reward_action_rate_penalty():
    """A jump in the control from the previous step is penalised (anti-chatter)."""
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    state = _state(jnp.pi)
    action = jnp.array([1.0, -1.0])
    # Same state and action: holding the previous action steady beats a large jump.
    steady_r = _r(env, params, state, action, prev=action)          # |a - a_prev| = 0
    jerky_r = _r(env, params, state, action, prev=-action)          # large step change
    assert jerky_r < steady_r
