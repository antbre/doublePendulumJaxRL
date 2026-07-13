"""Analytical double-pendulum dynamics in pure JAX.

We use the standard manipulator equation

    M(q) qddot + C(q, qdot) qdot + G(q) = tau

with generalised coordinates ``q = [theta1, theta2]`` where

* ``theta1`` is the angle of the first link measured from the *downward* vertical
  (so ``theta1 = 0`` hangs down, ``theta1 = pi`` points up), and
* ``theta2`` is the angle of the second link measured *relative to the first link*
  (so ``theta2 = 0`` means the second link is aligned with the first).

This is the classic Acrobot/Spong parameterisation; it makes joint torques map
directly onto generalised forces, which is exactly what we need for the three
actuation modes. All functions are pure and jittable / vmappable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import EnvParams

# A physical-state array is ``[theta1, theta2, omega1, omega2]``.
Array = jax.Array


def mass_matrix(theta2: Array, p: EnvParams) -> Array:
    """Configuration-dependent 2x2 mass matrix M(q)."""
    c2 = jnp.cos(theta2)
    d11 = (
        p.m1 * p.lc1**2
        + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * p.l1 * p.lc2 * c2)
        + p.i1
        + p.i2
    )
    d12 = p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2) + p.i2
    d22 = p.m2 * p.lc2**2 + p.i2
    return jnp.array([[d11, d12], [d12, d22]])


def coriolis_gravity(state: Array, p: EnvParams) -> Array:
    """Combined Coriolis/centrifugal + gravity bias vector C(q,qdot) qdot + G(q)."""
    theta1, theta2, omega1, omega2 = state
    s2 = jnp.sin(theta2)

    # Coriolis / centrifugal terms.
    h1 = -p.m2 * p.l1 * p.lc2 * s2 * (omega2**2 + 2.0 * omega1 * omega2)
    h2 = p.m2 * p.l1 * p.lc2 * s2 * omega1**2

    # Gravity terms: derivatives of the potential energy wrt each coordinate.
    g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g * jnp.sin(theta1) \
        + p.m2 * p.lc2 * p.g * jnp.sin(theta1 + theta2)
    g2 = p.m2 * p.lc2 * p.g * jnp.sin(theta1 + theta2)

    return jnp.array([h1 + g1, h2 + g2])


def accelerations(state: Array, tau: Array, p: EnvParams) -> Array:
    """Angular accelerations qddot for a given full joint-torque vector tau (shape (2,))."""
    theta2 = state[1]
    omega = state[2:4]
    damping = jnp.array([p.b1, p.b2]) * omega
    rhs = tau - damping - coriolis_gravity(state, p)
    qddot = jnp.linalg.solve(mass_matrix(theta2, p), rhs)
    return qddot


def _deriv(state: Array, tau: Array, p: EnvParams) -> Array:
    """Time derivative of the physical state ``[theta1, theta2, omega1, omega2]``."""
    qddot = accelerations(state, tau, p)
    return jnp.concatenate([state[2:4], qddot])


def rk4_step(state: Array, tau: Array, dt: float, p: EnvParams) -> Array:
    """One classical Runge-Kutta 4 integration step (torque held constant)."""
    k1 = _deriv(state, tau, p)
    k2 = _deriv(state + 0.5 * dt * k1, tau, p)
    k3 = _deriv(state + 0.5 * dt * k2, tau, p)
    k4 = _deriv(state + dt * k3, tau, p)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(state: Array, tau: Array, p: EnvParams) -> Array:
    """Advance the physical state by one control step using ``n_substeps`` RK4 steps."""
    sub_dt = p.dt / p.n_substeps

    def body(s, _):
        return rk4_step(s, tau, sub_dt, p), None

    state, _ = jax.lax.scan(body, state, None, length=p.n_substeps)
    return state


def kinetic_energy(state: Array, p: EnvParams) -> Array:
    qdot = state[2:4]
    return 0.5 * qdot @ (mass_matrix(state[1], p) @ qdot)


def potential_energy(state: Array, p: EnvParams) -> Array:
    theta1, theta2 = state[0], state[1]
    return (
        -(p.m1 * p.lc1 + p.m2 * p.l1) * p.g * jnp.cos(theta1)
        - p.m2 * p.lc2 * p.g * jnp.cos(theta1 + theta2)
    )


def total_energy(state: Array, p: EnvParams) -> Array:
    return kinetic_energy(state, p) + potential_energy(state, p)


def link_positions(theta1: Array, theta2: Array, p: EnvParams):
    """Cartesian positions of the two joint ends, for forward kinematics / rendering.

    Returns ``(x1, y1, x2, y2)`` where index 1 is the elbow and index 2 is the tip.
    """
    x1 = p.l1 * jnp.sin(theta1)
    y1 = -p.l1 * jnp.cos(theta1)
    x2 = x1 + p.l2 * jnp.sin(theta1 + theta2)
    y2 = y1 - p.l2 * jnp.cos(theta1 + theta2)
    return x1, y1, x2, y2


def tip_height(theta1: Array, theta2: Array, p: EnvParams) -> Array:
    """Height of the pendulum tip, normalised to [-1, 1] (+1 == fully upright)."""
    _, _, _, y2 = link_positions(theta1, theta2, p)
    return y2 / (p.l1 + p.l2)
