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
    # Per actuated joint. Gravity torque near horizontal is ~20-30 N*m for these unit
    # masses/lengths, so the joint torque is still well below "just muscle it up" and the
    # swing-up requires energy pumping; but 5.0 made the swing-up too hard to *discover*
    # (training stalled at the bottom), whereas 10.0 is reliably learnable.
    max_torque: float = 10.0

    # --- episode ---
    max_steps_in_episode: int = 1000

    # --- reward weights ---
    #   e1 = wrap(theta1 - pi), e2 = wrap(theta2)   # angle error from the upright target
    #   ang_err2 = e1^2 + e2^2, vel = w1^2 + w2^2, ctrl = |a|^2
    #   near    = exp(-near_coef  * ang_err2)       # ~1 close to the target, ~0 far away
    #   attract = exp(-bonus_coef * ang_err2)       # sharp attractor around the target
    #   reward = + w_height  * tip_height           # dense swing-up driver over [-1, 1]
    #            - w_upright  * ang_err2            # quadratic reward for the upright pose
    #            - w_ctrl     * ctrl               # quadratic penalty on actuation
    #            + w_swing    * max(tip_height, 0) # extra reward for the tip above horizontal
    #            + w_bonus    * attract            # localized attractor at the upright target
    #            - near * w_vel * vel              # joint-velocity penalty, only near the target
    #            - w_smooth   * |a - a_prev|^2     # control-smoothness (anti-chatter) penalty
    #
    # Rationale for each term:
    #  * +w_height * tip_height is a DENSE height reward across the whole swing (-1 hanging,
    #    +1 upright). It is what actually guides the energy-pumping swing-up: every bit of
    #    extra height pays off immediately, so the agent is not stuck sitting at the bottom.
    #    Without it a position-only quadratic reward has a strong "hang still" local optimum.
    #  * -w_upright * ang_err2 is an LQR-style quadratic reward, maximal (0) exactly at the
    #    upright target [pi, 0] and increasingly negative as the pose drifts away.
    #  * -w_ctrl * ctrl is a quadratic penalty on the (normalised) joint torques, favouring
    #    efficient, low-effort control.
    #  * +w_swing * max(tip_height, 0) is an extra bonus that only switches on once the tip
    #    rises above the horizontal line (tip_height > 0), rewarding a completed swing-up.
    #  * +w_bonus * near is a sharp, localized attractor at the target so that catching and
    #    holding the balance pays off clearly (the flat -ang_err2 basin alone is too weak).
    #  * -near * w_vel * vel penalises joint angular velocity, but the `near` gate makes it
    #    ~0 during the swing-up (so the agent is free to pump energy) and ~1 near the target
    #    (so it settles instead of spinning through the equilibrium).
    #
    # NOTE: all weights are deliberately O(0.1-1) so the per-step reward (and the episode
    # return) stays O(1) x n_steps. rejax's PPO normalises observations and advantages but
    # NOT rewards; a ~10x larger reward (e.g. w_upright=1 => -pi^2 ~ -10/step) makes the
    # critic targets too large to fit under gradient clipping and training stalls.
    w_height: float = 0.5
    w_upright: float = 0.1
    w_ctrl: float = 0.001
    w_swing: float = 0.5
    w_bonus: float = 3.0
    bonus_coef: float = 5.0
    w_vel: float = 0.05
    near_coef: float = 5.0
    # Action-rate (control-smoothness) penalty: -w_smooth * |a - a_prev|^2. Suppresses
    # chattering by penalising fast changes in the control from one step to the next.
    w_smooth: float = 0.05

    # --- reset / reference-state initialization (RSI) ---
    reset_noise: float = 0.05  # std of the near-bottom start perturbation
    # A fraction of episodes start near the *upright* target instead of hanging, so
    # the agent gets direct practice catching/balancing at the top (set 0.0 for pure
    # swing-up-from-bottom). Top starts are spread over a band of angles/velocities.
    p_start_top: float = 0.2
    top_angle_std: float = 0.5
    top_vel_std: float = 2.0
