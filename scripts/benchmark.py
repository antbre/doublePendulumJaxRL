#!/usr/bin/env python3
"""Benchmark CPU vs GPU throughput for env rollouts and PPO training.

JAX selects its backend at import time, so each platform is measured in a
fresh subprocess (``JAX_PLATFORMS=cpu`` vs default CUDA when available).

Examples
--------
    python scripts/benchmark.py
    python scripts/benchmark.py --timesteps 262144 --repeats 5
    python scripts/benchmark.py --platform cpu   # single platform only
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# worker (runs in a subprocess with JAX_PLATFORMS set before import)
# --------------------------------------------------------------------------- #
def _run_worker(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    from double_pendulum_jaxrl.config import ActuationMode
    from double_pendulum_jaxrl.env import DoublePendulum
    from double_pendulum_jaxrl.train import build_algo

    devices = [str(d) for d in jax.devices()]
    backend = jax.default_backend()

    def _time_call(fn, *call_args, warmup: int, repeats: int) -> List[float]:
        for _ in range(warmup):
            jax.block_until_ready(fn(*call_args))
        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn(*call_args)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return times

    # --- env rollout: vmap(num_envs) + scan(num_steps) ---
    env = DoublePendulum(ActuationMode.BOTH)
    params = env.default_params
    num_envs = args.num_envs
    num_steps = args.num_steps

    def _env_rollout(rng):
        reset_keys = jax.random.split(rng, num_envs)
        _obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
        actions = jnp.zeros((num_envs, env.act_dim))
        rng, scan_key = jax.random.split(rng)

        def body(carry, _):
            rng, state = carry
            rng, step_key = jax.random.split(rng)
            step_keys = jax.random.split(step_key, num_envs)
            _, state, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                step_keys, state, actions, params
            )
            return (rng, state), None

        (_, state), _ = jax.lax.scan(body, (scan_key, state), None, length=num_steps)
        return state.time

    env_fn = jax.jit(_env_rollout)
    env_times = _time_call(env_fn, jax.random.PRNGKey(args.seed), warmup=args.warmup, repeats=args.repeats)
    env_step_count = num_envs * num_steps

    # --- PPO training chunk (full jitted train loop, no eval callback) ---
    algo, config = build_algo(
        ActuationMode.BOTH,
        total_timesteps=args.timesteps,
        eval_freq=args.timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        skip_initial_evaluation=True,
    )
    train_fn = jax.jit(algo.train)
    train_times = _time_call(train_fn, jax.random.PRNGKey(args.seed), warmup=1, repeats=args.repeats)

    payload = {
        "platform": args.platform,
        "backend": backend,
        "devices": devices,
        "env": {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "env_steps": env_step_count,
            "times_s": env_times,
            "best_s": min(env_times),
            "median_s": statistics.median(env_times),
            "env_steps_per_s_best": env_step_count / min(env_times),
            "env_steps_per_s_median": env_step_count / statistics.median(env_times),
        },
        "ppo": {
            "timesteps": args.timesteps,
            "times_s": train_times,
            "best_s": min(train_times),
            "median_s": statistics.median(train_times),
            "env_steps_per_s_best": args.timesteps / min(train_times),
            "env_steps_per_s_median": args.timesteps / statistics.median(train_times),
        },
    }
    print(json.dumps(payload))


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def _spawn_worker(args: argparse.Namespace, platform: str) -> Optional[Dict[str, Any]]:
    env = os.environ.copy()
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env.pop("JAX_PLATFORMS", None)

    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--platform",
        platform,
        "--seed",
        str(args.seed),
        "--num-envs",
        str(args.num_envs),
        "--num-steps",
        str(args.num_steps),
        "--timesteps",
        str(args.timesteps),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{platform}] worker failed:", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        return None

    # Worker prints a single JSON line.
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _fmt_rate(steps: float, seconds: float) -> str:
    rate = steps / seconds
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f} M steps/s"
    if rate >= 1_000:
        return f"{rate / 1_000:.1f} k steps/s"
    return f"{rate:.0f} steps/s"


def _fmt_seconds(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _print_row(label: str, steps: int, best_s: float, median_s: float) -> None:
    print(
        f"  {label:<22}  best {_fmt_rate(steps, best_s):>14}  ({_fmt_seconds(best_s)})"
        f"   median {_fmt_rate(steps, median_s):>14}  ({_fmt_seconds(median_s)})"
    )


def _print_results(results: Dict[str, Dict[str, Any]]) -> None:
    print()
    print("=" * 72)
    print("Double-pendulum JAX RL — CPU vs GPU benchmark")
    print("=" * 72)

    for platform, data in results.items():
        print()
        print(f"[{platform.upper()}] backend={data['backend']}  devices={data['devices']}")
        env = data["env"]
        ppo = data["ppo"]
        _print_row("env rollout", env["env_steps"], env["best_s"], env["median_s"])
        _print_row("PPO train chunk", ppo["timesteps"], ppo["best_s"], ppo["median_s"])

    if "cpu" in results and "gpu" in results:
        print()
        print("Speedup (GPU / CPU, best-of repeats)")
        for bench_key, step_key, label in [
            ("env", "env_steps", "env rollout"),
            ("ppo", "timesteps", "PPO train chunk"),
        ]:
            cpu_best = results["cpu"][bench_key]["best_s"]
            gpu_best = results["gpu"][bench_key]["best_s"]
            print(f"  {label:<22}  {cpu_best / gpu_best:.2f}x")
        print(
            "\nNote: the env rollout micro-benchmark is tiny; CPU often wins due to GPU"
            "\nlaunch overhead. PPO train chunk is the meaningful training throughput metric."
        )
    print()


def _build_parser(*, worker: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Which backend(s) to benchmark (default: both).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=128,
        help="Parallel envs for rollout + PPO (default matches training).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Rollout horizon per PPO update (default matches training).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=131_072,
        help="Env steps for the PPO timing chunk (default: 8 PPO updates).",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (env only).")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per benchmark.")
    if worker:
        parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    worker_mode = "--worker" in sys.argv
    parser = _build_parser(worker=worker_mode)
    args = parser.parse_args()

    if worker_mode:
        _run_worker(args)
        return

    platforms: List[str]
    if args.platform == "both":
        platforms = ["cpu", "gpu"]
    else:
        platforms = [args.platform]

    print(
        f"Benchmark config: num_envs={args.num_envs}, num_steps={args.num_steps}, "
        f"ppo_timesteps={args.timesteps}, warmup={args.warmup}, repeats={args.repeats}"
    )

    results: Dict[str, Dict[str, Any]] = {}
    for platform in platforms:
        print(f"Running {platform} benchmark...", flush=True)
        data = _spawn_worker(args, platform)
        if data is None:
            if platform == "gpu":
                print(
                    "GPU benchmark unavailable (install with `pip install -e \".[cuda]\"`).",
                    file=sys.stderr,
                )
            continue
        results[platform] = data

    if not results:
        raise SystemExit("No benchmark results collected.")

    _print_results(results)


if __name__ == "__main__":
    main()
