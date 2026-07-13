# Double Pendulum JAX RL

A **pure-JAX** reinforcement-learning environment for the **double-pendulum swing-up + balance**
task. The dynamics are the analytical Lagrangian equations of motion integrated with RK4 — no
physics engine — so the whole environment is `jit`/`vmap`-friendly and runs entirely on-device
(**GPU if one is present, otherwise CPU**, with no code changes).

Training uses [`rejax`](https://github.com/keraJLi/rejax) (pure-JAX PPO); the environment follows
the [`gymnax`](https://github.com/RobertTLange/gymnax) `Environment` API.

## Actuation modes

The environment can be configured three ways (each with a per-joint `max_torque`):

| Mode          | CLI value | Description                                    | Action dim |
|---------------|-----------|------------------------------------------------|-----------|
| `BOTH`        | `both`    | Fully actuated — both joints driven            | 2         |
| `TOP_ONLY`    | `top`     | Only the shoulder joint driven ("pendubot")    | 1         |
| `BOTTOM_ONLY` | `bottom`  | Only the elbow joint driven (classic "acrobot")| 1         |

Angles are measured from the **downward** vertical: `theta1 = theta2 = 0` hangs straight down,
and the upright balance target is `theta1 = pi, theta2 = 0`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # CPU
# On a machine with an NVIDIA GPU, also run:
pip install -e ".[cuda]"   # pulls the CUDA jaxlib; same code, GPU auto-detected
```

Check which device JAX will use:

```bash
python -c "import jax; print(jax.devices())"
```

> This repository was developed and verified on a **CPU-only** machine (no CUDA GPU present).
> The code is device-agnostic: on a CUDA machine JAX picks up the GPU automatically.

## Train

```bash
python scripts/train.py --mode both                 # default budget
python scripts/train.py --mode top --timesteps 5000000 --seed 1
```

This writes a policy checkpoint to `checkpoints/<mode>.pkl` and a learning-curve PNG to
`checkpoints/<mode>_curve.png`.

### Following progress during training

The whole PPO loop is compiled into one `jax.jit` call, so a plain `print` can't report
mid-run. Training instead **streams a progress line at every evaluation** (host-side
`jax.debug.callback`):

```
step=     16384  return= -618.63 +/-  60.64  ep_len= 500.0
step=     32768  return= -591.47 +/-  50.85  ep_len= 500.0
```

- Use `--eval-freq N` to print more/less often (it is rounded up to a multiple of
  `num_envs * num_steps`; default `eval_freq` is 25000).
- Use `--quiet` to turn the streaming lines off.
- The full curve is always saved as a PNG at the end regardless.
- To watch hardware utilisation alongside it: `htop` (CPU) or `nvidia-smi -l 1` (GPU).

## Test / visualize a trained policy

```bash
python scripts/play.py --mode both --save both.gif --diagnostics both_diag.png
python scripts/play.py --checkpoint checkpoints/top.pkl        # live window
```

`play.py` rolls out one episode, animates the pendulum, and (optionally) saves a diagnostics plot
of angles, velocities, torques, tip height and reward.

## Run the tests

```bash
pip install pytest
pytest
```

Covers energy conservation of the unactuated pendulum (RK4), mass-matrix positive-definiteness,
gravity equilibria, the gymnax API contract, `jit`/`vmap`-ability, determinism, and torque routing
per actuation mode.

## Project layout

```
double_pendulum_jaxrl/
├── config.py       # ActuationMode enum + EnvParams (physics + reward)
├── dynamics.py     # analytical M/C/G equations of motion + RK4 integrator
├── env.py          # DoublePendulum gymnax Environment
├── train.py        # rejax PPO configuration + training
├── evaluate.py     # checkpoint save/load + policy rollout
└── visualize.py    # matplotlib animation + diagnostics + learning curve
scripts/
├── train.py        # CLI: train a mode
└── play.py         # CLI: load a checkpoint and render
tests/              # pytest suite
```

## Notes on training

- `BOTH` (fully actuated) learns the swing-up most reliably.
- `TOP_ONLY` and `BOTTOM_ONLY` are genuinely hard **underactuated** swing-up problems; they get a
  larger default timestep budget and may still need hyperparameter tuning (in `default_ppo_config`
  in `double_pendulum_jaxrl/train.py`) to fully stabilise inverted.
- Reward weights (`w_up`, `w_vel`, `w_ctrl`) and physical parameters live in `EnvParams`
  (`config.py`) and can be tuned without touching the dynamics.
