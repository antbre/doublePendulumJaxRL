#!/usr/bin/env python3
"""CLI: train a double-pendulum swing-up policy for a chosen actuation mode.

Examples
--------
    python scripts/train.py --mode both
    python scripts/train.py --mode top --timesteps 5_000_000 --seed 1
"""

from __future__ import annotations

import argparse
import os

import jax

from double_pendulum_jaxrl.config import ActuationMode
from double_pendulum_jaxrl.evaluate import save_checkpoint
from double_pendulum_jaxrl.train import train
from double_pendulum_jaxrl.visualize import plot_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "top", "bottom"], default="both")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total_timesteps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="Checkpoint path (default checkpoints/<mode>.pkl).")
    parser.add_argument("--curve", type=str, default=None,
                        help="Where to save the learning-curve PNG.")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Env steps between progress prints/evals (default 25000).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the streaming per-eval progress lines.")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    mode = ActuationMode.from_str(args.mode)

    overrides = {}
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.eval_freq is not None:
        overrides["eval_freq"] = args.eval_freq

    print(f"Training mode={mode.name} seed={args.seed} ...")
    algo, train_state, evaluation, config = train(
        mode, seed=args.seed, verbose=not args.quiet, **overrides
    )

    _lengths, returns = evaluation
    print(f"Final mean return: {float(returns[-1].mean()):.2f}")

    out = args.out or os.path.join("checkpoints", f"{args.mode}.pkl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_checkpoint(out, mode, config, train_state)
    print(f"Saved checkpoint -> {out}")

    curve = args.curve or os.path.join("checkpoints", f"{args.mode}_curve.png")
    plot_learning_curve(evaluation, config["eval_freq"], save_path=curve)
    print(f"Saved learning curve -> {curve}")


if __name__ == "__main__":
    main()
