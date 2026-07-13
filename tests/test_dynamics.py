"""Physics correctness tests for the double-pendulum dynamics."""

import jax
import jax.numpy as jnp
import numpy as np

from double_pendulum_jaxrl.config import EnvParams
from double_pendulum_jaxrl import dynamics

jax.config.update("jax_enable_x64", True)  # tighter energy conservation for the test


def test_energy_conservation_unactuated():
    """With no torque and no damping, total mechanical energy is conserved."""
    p = EnvParams(b1=0.0, b2=0.0, dt=0.01, n_substeps=10)
    state = jnp.array([0.5, -0.3, 0.4, -0.2])  # arbitrary non-trivial start
    tau = jnp.zeros(2)

    e0 = dynamics.total_energy(state, p)

    def body(s, _):
        return dynamics.integrate(s, tau, p), dynamics.total_energy(s, p)

    _, energies = jax.lax.scan(body, state, None, length=2000)
    drift = np.abs(np.asarray(energies) - float(e0))
    assert drift.max() < 1e-3, f"energy drift too large: {drift.max()}"


def test_mass_matrix_symmetric_pd():
    p = EnvParams()
    for th2 in np.linspace(-np.pi, np.pi, 9):
        M = np.asarray(dynamics.mass_matrix(jnp.asarray(th2), p))
        assert np.allclose(M, M.T), "mass matrix must be symmetric"
        eigs = np.linalg.eigvalsh(M)
        assert (eigs > 0).all(), "mass matrix must be positive definite"


def test_gravity_equilibria():
    """Bias vector (Coriolis+gravity) vanishes hanging down and pointing up (at rest)."""
    p = EnvParams()
    down = jnp.array([0.0, 0.0, 0.0, 0.0])
    up = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
    assert np.allclose(np.asarray(dynamics.coriolis_gravity(down, p)), 0.0, atol=1e-6)
    assert np.allclose(np.asarray(dynamics.coriolis_gravity(up, p)), 0.0, atol=1e-6)


def test_tip_height_extremes():
    p = EnvParams()
    assert float(dynamics.tip_height(jnp.pi, 0.0, p)) == 1.0   # fully upright
    assert float(dynamics.tip_height(0.0, 0.0, p)) == -1.0     # hanging straight down
