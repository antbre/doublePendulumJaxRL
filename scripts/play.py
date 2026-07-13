#!/usr/bin/env python3
"""CLI: load a trained policy, roll out one episode, and visualise it.

Examples
--------
    python scripts/play.py --mode both --save both.gif
    python scripts/play.py --checkpoint checkpoints/top.pkl --diagnostics top_diag.png
"""

from __future__ import annotations

import argparse
import os

import jax

from double_pendulum_jaxrl.evaluate import load_policy, rollout
from double_pendulum_jaxrl.visualize import animate, plot_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "top", "bottom"], default=None,
                        help="Convenience: load checkpoints/<mode>.pkl.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None,
                        help="Save animation to this path (.gif/.mp4). Omit to show live.")
    parser.add_argument("--diagnostics", type=str, default=None,
                        help="Save a diagnostics plot to this PNG path.")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        if args.mode is None:
            parser.error("Provide either --checkpoint or --mode.")
        ckpt = os.path.join("checkpoints", f"{args.mode}.pkl")

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading policy from {ckpt} ...")
    act, env, env_params = load_policy(ckpt)

    traj = rollout(act, env, env_params, seed=args.seed)
    print(f"Episode return: {traj['reward'].sum():.2f} | "
          f"final tip height: {traj['tip_height'][-1]:.3f}")

    if args.diagnostics:
        plot_diagnostics(traj, env_params, save_path=args.diagnostics)
        print(f"Saved diagnostics -> {args.diagnostics}")

    animate(traj, env_params, save_path=args.save)
    if args.save:
        print(f"Saved animation -> {args.save}")


if __name__ == "__main__":
    main()
