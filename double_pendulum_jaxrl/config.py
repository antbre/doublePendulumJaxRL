"""Configuration for the double-pendulum environment.

Everything that parameterises the *physics* and the *reward* lives in ``EnvParams``,
a ``flax.struct.dataclass`` so it is a JAX pytree and can be traced / vmapped.

The one thing that is *not* a traced value is the actuation mode: it decides the
size of the action space, which must be a static Python value, so it is passed to
the environment constructor rather than living in ``EnvParams``.
"""

from __future__ import annotations

from enum import IntEnum

from flax import struct


class ActuationMode(IntEnum):
    """Which joints the agent is allowed to drive."""

    BOTH = 0        # fully actuated: torque on shoulder (joint 0) and elbow (joint 1)
    TOP_ONLY = 1    # only the shoulder joint is actuated ("pendubot")
    BOTTOM_ONLY = 2  # only the elbow joint is actuated (classic "acrobot")

    @classmethod
    def from_str(cls, name: str) -> "ActuationMode":
        return {
            "both": cls.BOTH,
            "top": cls.TOP_ONLY,
            "top_only": cls.TOP_ONLY,
            "bottom": cls.BOTTOM_ONLY,
            "bottom_only": cls.BOTTOM_ONLY,
        }[name.lower()]

    @property
    def action_dim(self) -> int:
        return 2 if self == ActuationMode.BOTH else 1


@struct.dataclass
class EnvParams:
    """Physical + task parameters for the double pendulum.

    Angles are measured so that ``theta1 = theta2 = 0`` is the pendulum hanging
    straight down, and the upright (balance) target is ``theta1 = pi, theta2 = 0``.
    """

    # --- link masses (point masses at the end of each link) ---
    m1: float = 1.0
    m2: float = 1.0
    # --- link lengths ---
    l1: float = 1.0
    l2: float = 1.0
    # --- centre-of-mass distance along each link (point mass => = length) ---
    lc1: float = 1.0
    lc2: float = 1.0
    # --- link moments of inertia about the com (point mass => 0) ---
    i1: float = 0.0
    i2: float = 0.0
    # --- gravity ---
    g: float = 9.81
    # --- joint viscous damping ---
    b1: float = 0.0
    b2: float = 0.0

    # --- integration ---
    dt: float = 0.02          # control timestep (s)
    # RK4 substeps per control step; static because it is used as a scan length.
    n_substeps: int = struct.field(pytree_node=False, default=4)

    # --- actuation ---
    max_torque: float = 5.0   # per actuated joint

    # --- episode ---
    max_steps_in_episode: int = 500

    # --- reward weights ---
    w_up: float = 1.0         # reward for tip height (upright = +1)
    w_vel: float = 0.02       # penalty on angular velocity magnitude
    w_ctrl: float = 0.002     # penalty on control effort
    reset_noise: float = 0.05  # std of initial angle/velocity perturbation
