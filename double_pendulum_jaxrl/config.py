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
    max_steps_in_episode: int = 1000

    # --- reward weights ---
    #   e1 = wrap(theta1 - pi), e2 = wrap(theta2), vel = w1^2 + w2^2, ctrl = |a|^2
    #   gate = ((tip_height + 1) / 2) ** stab_power     # ~0 at bottom, ~1 near the top
    #   reward = w_up * tip_height
    #            - gate * (w_vel*vel + w_ctrl*ctrl + w_ang*(e1^2 + e2^2))
    #            - w_smooth * |a - a_prev|^2
    #            + w_bonus * exp(-(bonus_ang_coef*(e1^2 + e2^2) + bonus_vel_coef*vel))
    #
    # This is the version that was verified to swing up AND catch upright (it still
    # oscillates/chatters around the equilibrium, to be addressed separately):
    #  * w_up * tip_height drives the energy-pumping swing-up.
    #  * the gated velocity/control/angle penalties activate near the top so the agent
    #    catches and regulates upright instead of swinging through.
    #  * the smooth bonus provides a sharp basin at the exact equilibrium.
    #  * the action-rate term suppresses chattering on the control signal.
    w_up: float = 1.0
    w_vel: float = 0.02
    w_ctrl: float = 0.002
    w_ang: float = 1.0
    w_smooth: float = 0.1
    stab_power: float = 3.0
    w_bonus: float = 5.0
    bonus_ang_coef: float = 5.0
    bonus_vel_coef: float = 0.1

    # --- reset / reference-state initialization (RSI) ---
    reset_noise: float = 0.05  # std of the near-bottom start perturbation
    # A fraction of episodes start near the *upright* target instead of hanging, so
    # the agent gets direct practice catching/balancing at the top (set 0.0 for pure
    # swing-up-from-bottom). Top starts are spread over a band of angles/velocities.
    p_start_top: float = 0.2
    top_angle_std: float = 0.5
    top_vel_std: float = 2.0
