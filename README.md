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

### Periodic checkpoints & resuming

Save a full, **resumable** checkpoint every N env steps, then continue later:

```bash
# checkpoint every 500k steps -> checkpoints/both_step<N>.pkl (+ final checkpoints/both.pkl)
python scripts/train.py --mode both --checkpoint-every 500000

# continue an earlier run for 1M ADDITIONAL steps (mode is taken from the checkpoint)
python scripts/train.py --resume checkpoints/both_step500000.pkl --timesteps 1000000
```

Checkpoints store the full rejax train state (actor, critic, optimizer, RNG, step
count) so resuming continues the exact optimization — as well as the actor params that
`play.py` needs. Since the training loop is a single jitted call, periodic checkpointing
runs it in Python-level chunks of `--checkpoint-every` steps.

### Following progress during training

The whole PPO loop is compiled into one `jax.jit` call, so a plain `print` can't report
mid-run. Training instead **streams a progress line at every evaluation** (host-side
`jax.debug.callback`):

```
step=    131072  return=-1054.22 +/- 220.09  ep_len=1000.0
step=   1048576  return= -186.56 +/- 157.95  ep_len=1000.0
```

(Return climbs from roughly −1500 at the bottom toward positive values once the policy
swings up and balances; see `## Notes on training` for the reward scale.)

- Use `--eval-freq N` to print more/less often (it is rounded up to a multiple of
  `num_envs * num_steps`; default `eval_freq` is 25000).
- Use `--quiet` to turn the streaming lines off.
- The full curve is always saved as a PNG at the end regardless.
- To watch hardware utilisation alongside it: `htop` (CPU) or `nvidia-smi -l 1` (GPU).

## Pretrained baselines

Ready-to-run baseline policies for all three modes are checked into `checkpoints/`, so you can
visualize a swing-up without training anything first:

| Checkpoint               | Mode          | Learning curve                 |
|--------------------------|---------------|--------------------------------|
| `checkpoints/both.pkl`   | `BOTH`        | `checkpoints/both_curve.png`   |
| `checkpoints/top.pkl`    | `TOP_ONLY`    | `checkpoints/top_curve.png`    |
| `checkpoints/bottom.pkl` | `BOTTOM_ONLY` | `checkpoints/bottom_curve.png` |

Each `.pkl` holds the full rejax train state (actor, critic, optimizer, RNG, step count), so it
can be rendered with `play.py`, resumed with `--resume`, or used as a warm start; the matching
`_curve.png` is the training-return curve. Play one directly:

```bash
python scripts/play.py --mode both          # loads checkpoints/both.pkl
```

Re-running `scripts/train.py --mode <mode>` overwrites the corresponding files with a fresh run.

## Test / visualize a trained policy

```bash
python scripts/play.py --mode both --save both.gif --diagnostics both_diag.png
python scripts/play.py --checkpoint checkpoints/top.pkl        # live window
```

`play.py` rolls out one episode, animates the pendulum, and (optionally) saves a diagnostics plot
of angles, velocities, torques, tip height and reward. By default it evaluates the **deterministic
mean** policy; pass `--stochastic` to sample from the policy instead (rejax's training-time policy
has std ≈ 1, so the stochastic rollout looks like bang-bang chatter — the mean is the real
learned controller).

## Run the tests

```bash
pip install pytest
pytest
```

Covers energy conservation of the unactuated pendulum (RK4), mass-matrix positive-definiteness,
gravity equilibria, the gymnax API contract, `jit`/`vmap`-ability, determinism, torque routing
per actuation mode, and the reward-shaping terms (upright/actuation, dense height driver, swing
bonus, gated velocity penalty, and the action-rate anti-chatter penalty).

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
- The reward is an LQR-style shaping (see the formula in `EnvParams`): a dense `w_height`
  swing-up driver, a quadratic `w_upright` upright reward, a quadratic `w_ctrl` actuation
  penalty, a `w_swing` bonus above horizontal, a sharp `w_bonus` attractor at the target, a
  `w_vel` joint-velocity penalty gated to activate only near the target, and a `w_smooth`
  control-smoothness penalty on `|a - a_prev|^2` that suppresses actuator chatter. Weights are
  kept O(0.1–1) on purpose — rejax normalises observations and advantages but *not* rewards, so
  an oversized reward makes the critic targets unfittable and training stalls.
- `max_torque` (per joint) defaults to `10.0`: still well below the ~20–30 N·m of gravity torque
  (so the swing-up requires energy pumping), but large enough that the swing-up is *discoverable*
  — `5.0` was too weak and training stalled at the bottom.
- Training uses reference-state initialization: a fraction of episodes (`p_start_top`) start near
  the upright target so the agent gets direct practice catching/balancing; evaluation always
  starts from the bottom, so the reported return is an honest swing-up-from-hanging measure.
- Reward weights and physical parameters live in `EnvParams` (`config.py`) and can be tuned
  without touching the dynamics.
