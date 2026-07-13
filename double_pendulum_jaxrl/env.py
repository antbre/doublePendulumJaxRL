"""Gymnax-compatible double-pendulum swing-up + balance environment.

The environment subclasses ``gymnax.environments.environment.Environment`` so it
plugs directly into ``rejax`` (and any other gymnax-consuming trainer). The whole
step is pure JAX, so ``rejax`` can ``jit``/``vmap`` thousands of copies on-device.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from . import dynamics
from .config import ActuationMode, EnvParams


@struct.dataclass
class EnvState(environment.EnvState):
    theta1: jax.Array
    theta2: jax.Array
    omega1: jax.Array
    omega2: jax.Array
    time: int


class DoublePendulum(environment.Environment[EnvState, EnvParams]):
    """Double pendulum with selectable joint actuation.

    Parameters
    ----------
    actuation_mode:
        Which joints may be driven. This is a *static* Python value because it
        determines the size of the action space.
    """

    def __init__(self, actuation_mode: ActuationMode | str | int = ActuationMode.BOTH):
        super().__init__()
        if isinstance(actuation_mode, str):
            actuation_mode = ActuationMode.from_str(actuation_mode)
        self.actuation_mode = ActuationMode(int(actuation_mode))
        self.act_dim = self.actuation_mode.action_dim

    # ------------------------------------------------------------------ #
    # gymnax API
    # ------------------------------------------------------------------ #
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _action_to_torque(self, action: jax.Array, params: EnvParams) -> jax.Array:
        """Map the (bounded) agent action to a full 2-vector of joint torques."""
        a = jnp.clip(jnp.atleast_1d(action), -1.0, 1.0)
        if self.actuation_mode == ActuationMode.BOTH:
            tau = jnp.array([a[0], a[1]])
        elif self.actuation_mode == ActuationMode.TOP_ONLY:
            tau = jnp.array([a[0], 0.0])
        else:  # BOTTOM_ONLY
            tau = jnp.array([0.0, a[0]])
        return params.max_torque * tau

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        tau = self._action_to_torque(action, params)

        phys = jnp.array([state.theta1, state.theta2, state.omega1, state.omega2])
        phys = dynamics.integrate(phys, tau, params)

        new_state = EnvState(
            theta1=phys[0],
            theta2=phys[1],
            omega1=phys[2],
            omega2=phys[3],
            time=state.time + 1,
        )

        reward = self._reward(new_state, action, params)
        done = self.is_terminal(new_state, params)
        info = {
            "tip_height": dynamics.tip_height(new_state.theta1, new_state.theta2, params),
            "discount": self.discount(new_state, params),
        }
        return self.get_obs(new_state), new_state, reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        # Start hanging down (theta1 = theta2 = 0) with a small random perturbation.
        noise = params.reset_noise * jax.random.normal(key, (4,))
        state = EnvState(
            theta1=noise[0],
            theta2=noise[1],
            omega1=noise[2],
            omega2=noise[3],
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(
        self, state: EnvState, params: Optional[EnvParams] = None, key=None
    ) -> jax.Array:
        # Trig-encode the angles so the policy never sees a 2*pi wraparound jump.
        return jnp.array(
            [
                jnp.cos(state.theta1),
                jnp.sin(state.theta1),
                jnp.cos(state.theta2),
                jnp.sin(state.theta2),
                state.omega1,
                state.omega2,
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        # Swing-up has no failure state; episodes end purely on the time limit.
        return state.time >= params.max_steps_in_episode

    # ------------------------------------------------------------------ #
    # reward
    # ------------------------------------------------------------------ #
    def _reward(self, state: EnvState, action: jax.Array, params: EnvParams) -> jax.Array:
        up = dynamics.tip_height(state.theta1, state.theta2, params)  # in [-1, 1]
        vel_pen = state.omega1**2 + state.omega2**2
        ctrl_pen = jnp.sum(jnp.square(jnp.atleast_1d(action)))
        return params.w_up * up - params.w_vel * vel_pen - params.w_ctrl * ctrl_pen

    # ------------------------------------------------------------------ #
    # spaces / metadata
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"DoublePendulum-{self.actuation_mode.name}"

    @property
    def num_actions(self) -> int:
        return self.act_dim

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=jnp.float32)

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        high = jnp.array([1.0, 1.0, 1.0, 1.0, jnp.inf, jnp.inf], dtype=jnp.float32)
        return spaces.Box(low=-high, high=high, shape=(6,), dtype=jnp.float32)

    def state_space(self, params: Optional[EnvParams] = None) -> spaces.Dict:
        return spaces.Dict(
            {
                "theta1": spaces.Box(-jnp.inf, jnp.inf, (), jnp.float32),
                "theta2": spaces.Box(-jnp.inf, jnp.inf, (), jnp.float32),
                "omega1": spaces.Box(-jnp.inf, jnp.inf, (), jnp.float32),
                "omega2": spaces.Box(-jnp.inf, jnp.inf, (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode if params else 1),
            }
        )
