"""Matplotlib visualisation: animate a rollout and plot diagnostics.

Uses the ``Agg`` backend automatically when only saving to a file, so it works on
headless machines.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import EnvParams


def _forward_kinematics(theta1: np.ndarray, theta2: np.ndarray, p: EnvParams):
    x1 = p.l1 * np.sin(theta1)
    y1 = -p.l1 * np.cos(theta1)
    x2 = x1 + p.l2 * np.sin(theta1 + theta2)
    y2 = y1 - p.l2 * np.cos(theta1 + theta2)
    return x1, y1, x2, y2


def _torque_arrow(center, torque, max_torque, r_scale, color="gold"):
    """A rounded (curved) yellow arrow around a joint.

    Its radius is proportional to ``|torque| / max_torque`` and it sweeps
    counter-clockwise for positive torque, clockwise for negative. Returns a
    ``FancyArrowPatch`` (in data coordinates) or ``None`` if the torque is ~0.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.transforms as mtransforms

    mag = abs(float(torque)) / max(float(max_torque), 1e-6)
    if mag < 0.02:
        return None
    r = mag * r_scale                      # radius encodes the torque norm
    sweep, start = 270.0, -90.0
    arc = Path.arc(start, start + sweep)   # unit CCW arc
    if torque < 0.0:                       # clockwise: reverse so the head leads the other way
        arc = Path(arc.vertices[::-1], arc.codes)
    trans = mtransforms.Affine2D().scale(r).translate(center[0], center[1])
    return FancyArrowPatch(
        path=arc.transformed(trans),
        arrowstyle="-|>",
        mutation_scale=13,
        color=color,
        lw=2.2,
        zorder=5,
    )


def animate(traj: Dict[str, np.ndarray], params: EnvParams, save_path: Optional[str] = None,
            fps: int = 50, stride: int = 1):
    """Animate a rollout. If ``save_path`` is given, render to GIF/MP4; else show live."""
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    theta1 = traj["theta1"][::stride]
    theta2 = traj["theta2"][::stride]
    x1, y1, x2, y2 = _forward_kinematics(theta1, theta2, params)

    # Per-joint applied torque [shoulder, elbow] for the torque arrows.
    torque = traj.get("torque")
    if torque is not None:
        torque = np.asarray(torque)[::stride].reshape(len(theta1), -1)
    r_scale = 0.4 * params.l1  # full torque -> arc radius ~0.4 * link length

    reach = params.l1 + params.l2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.1 * reach, 1.1 * reach)
    ax.set_ylim(-1.1 * reach, 1.1 * reach)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(reach, color="green", lw=0.8, ls="--", alpha=0.5)  # upright target height
    ax.set_title("Double pendulum (yellow arrows = joint torque)")

    (line,) = ax.plot([], [], "o-", lw=3, markersize=8, color="tab:blue")
    trace, = ax.plot([], [], "-", lw=1, alpha=0.3, color="tab:orange")
    trace_x, trace_y = [], []
    arrows = []  # live torque-arrow patches, rebuilt each frame

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace

    def update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        trace_x.append(x2[i])
        trace_y.append(y2[i])
        trace.set_data(trace_x, trace_y)

        for a in arrows:
            a.remove()
        arrows.clear()
        if torque is not None:
            centers = [(0.0, 0.0), (x1[i], y1[i])]  # shoulder joint, elbow joint
            for j, c in enumerate(centers):
                if j < torque.shape[1]:
                    arr = _torque_arrow(c, torque[i, j], params.max_torque, r_scale)
                    if arr is not None:
                        ax.add_patch(arr)
                        arrows.append(arr)
        return line, trace, *arrows

    # blit=False so dynamically added/removed arrow patches render correctly.
    anim = FuncAnimation(fig, update, frames=len(theta1), init_func=init,
                         interval=1000 / fps, blit=False)

    if save_path is not None:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, fps=fps)
        plt.close(fig)
        return save_path
    plt.show()
    return anim


def plot_diagnostics(traj: Dict[str, np.ndarray], params: EnvParams,
                     save_path: Optional[str] = None):
    """Plot angles, velocities, torque and reward over the episode."""
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(traj["theta1"])) * params.dt
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(t, np.unwrap(traj["theta1"]), label=r"$\theta_1$")
    axes[0].plot(t, np.unwrap(traj["theta2"]), label=r"$\theta_2$")
    axes[0].set_ylabel("angle [rad]")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, traj["omega1"], label=r"$\omega_1$")
    axes[1].plot(t, traj["omega2"], label=r"$\omega_2$")
    axes[1].set_ylabel("ang. vel [rad/s]")
    axes[1].legend(loc="upper right")

    action = np.atleast_2d(traj["action"].reshape(len(t), -1).T)
    for j, a in enumerate(action):
        axes[2].plot(t, a, label=f"action[{j}]")
    axes[2].set_ylabel("action")
    axes[2].legend(loc="upper right")

    axes[3].plot(t, traj["tip_height"], color="tab:green", label="tip height (+1=up)")
    axes[3].plot(t, traj["reward"], color="tab:red", alpha=0.6, label="reward")
    axes[3].axhline(1.0, color="green", ls="--", lw=0.8, alpha=0.5)
    axes[3].set_ylabel("reward / height")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    plt.show()
    return fig


def plot_learning_curve(evaluation, eval_freq: int, save_path: Optional[str] = None):
    """Plot mean episodic return vs environment steps from rejax's evaluation output."""
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _lengths, returns = evaluation
    returns = np.asarray(returns)
    mean_return = returns.mean(axis=-1)
    std_return = returns.std(axis=-1)
    steps = np.arange(len(mean_return)) * eval_freq

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, mean_return, color="tab:blue")
    ax.fill_between(steps, mean_return - std_return, mean_return + std_return,
                    alpha=0.2, color="tab:blue")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean episodic return")
    ax.set_title("PPO learning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    plt.show()
    return fig
