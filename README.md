# Neural Residual Controller

A from-scratch implementation of a neural residual controller robust to plant/model mismatch. Trained end-to-end via backpropagation through differentiable physics (MJX) under domain randomization. Compares three architectures on a swing-up task.

## Formulation

The control law applied to the plant at every step:

```
u(t) = u*(t) + PD(x*(t) - x(t)) + v(t)
```

where `u*` and `x*` come from offline trajectory optimization on the _nominal_ plant, `PD` is a fixed feedback term, and `v(t)` is a learned residual whose purpose is to correct for the gap between the nominal plant (used by TO) and the actual plant.

Three options for `v(t)` are compared:

- **PD-only**: `v(t) = 0` (no residual)
- **Pure MLP**: `v(t) = MLP(history, reference)`
- **Two-model**: `v(t) = MLP(history, reference, θ_hat)` where `θ_hat = estimator(history)`

Pure MLP is the natural baseline ("just give the network everything and let it figure out the plant").
Two-model decouples the problem: a separate frozen estimator does
system identification from observed `(x, u)` history, and the controller gets `θ_hat` as an explicit input.

## Domain randomization

Each rollout samples a fresh plant from `cfg.DR_RANGES`:

- `mass_scale` per link — multiplies nominal body mass
- `damping` per link — applied to each DoF
- `frictionloss` per link — Coulomb friction on each DoF

Theta is applied to the MJX model in-place via `apply_theta` before each rollout.

The estimator's input is **history only** — `(x_hist, u_hist)`. References carry no information about theta and are excluded.

## Truncated BPTT

Trajectories from `solve_trajectory.py` have horizon `T = SIM_DURATION / TIMESTEP` steps (1000 for the double pendulum at 2 s, 2 ms). Backpropagating through all `T` MJX steps per iteration is impractical: memory grows linearly with `T`, and gradient norms can explode through long unrolls of stiff dynamics.

Instead, each training iteration samples a random window of length `n_rollout = H` (= 100) from a random TO trajectory and a random start index `t0`. The rollout starts at the reference state `x_refs[idx, t0]`, runs for `H` MJX steps under the closed-loop law, and the loss is computed over the resulting `H+1` states (the `H` rolled-out states plus the post-step terminal state):

```
tracking_loss(xs_full, x_refs[idx, t0:t0+H+1], nq) + alpha_reg * reg_loss(vs)
```

Each iteration sees a different combination of (trajectory, start time, plant) so over many iterations the controller is exposed to every segment of the swing-up under a wide distribution of plants. The history buffers (`x_hist0`, `u_hist0`, ...) are initialized by padding with `x_refs[idx, t0]` — a simplification that matches what deployment will see in its first few steps.

At evaluation (`scripts/evaluate.py`), rollouts run the **full `T` steps** to test whether each controller actually completes the task end-to-end.

## Pipeline

Core run order — `train/` produces artifacts, `eval/` consumes them:

```
python train/solve_trajectory.py         # → trajectories.npz
python train/train_pure.py               # → pure_params.pkl
python train/train_theta_estimator.py    # → theta_params.pkl
python train/train_controller.py         # → controller_params.pkl
python eval/evaluate.py                  # → metrics.npz   (pd / pure / two_model)
python eval/plot_final.py                # → training_curves.png, eval_metrics.png
```

Additional eval / diagnostic scripts, run as needed:

```
python eval/plot_trajectories.py         # sanity-check the TO references (no GPU/MuJoCo)
python eval/plot_eval_rollouts.py        # closed-loop rollout viz, per plant
python eval/evaluate_pure.py             # full pd-vs-pure metrics + |v|rms → metrics_pure.npz
python eval/evaluate_estimator.py        # per-parameter θ identifiability (R²)
```

Each training script reads `trajectories.npz` and the relevant `cfg` dict
(`cfg.PURE`, `cfg.THETA`, `cfg.CONTROLLER`). All outputs land in
`outputs/double_pendulum/`.

## Porting to a different plant

Plant-specific code is isolated in `<plant>/` directories. To port to a new robot:

1. **Create `<plant>/model.xml`** (MJCF). Must be Pinocchio-compatible: no Mujoco-only
   Features Pinocchio's `buildModelFromMJCF` rejects (mesh decompositions, exotic actuator types). Set `contype=0 conaffinity=0` to disable contact if contact-free training is desired.

2. **Create `<plant>/config.py`** with at minimum:
   - `MODEL_PATH`, `NQ`, `NV`, `NU`, `N_LINKS`
   - `TIMESTEP`, `SIM_DURATION` (often needs retuning — fast plants need smaller dt)
   - `KP`, `KD` — per-DoF feedback gains
   - `INITIAL_QPOS_RANGE`, `TARGET_QPOS_RANGE`, `N_TRAJECTORIES`
   - `TO_COST_X_RUNNING`, `TO_COST_U_RUNNING`, `TO_COST_X_TERMINAL` — TO weights; usually need retuning per plant
   - `DR_RANGES` and `THETA_DIM` — if the DR structure changes (e.g., different per-link parameters), also edit `lib/domain_randomization.py`'s `sample_theta` / `apply_theta`
   - `PURE`, `THETA`, `CONTROLLER` hyperparameter dicts

3. **Switch imports in `train/*.py` and `eval/*.py`** — change `from double_pendulum import config as cfg`
   to `from <plant> import config as cfg`.

4. **Things in `lib/` that should NOT need changing**: networks, rollout, losses are
   plant-agnostic and only depend on dimensions from the config.

## Repo layout

```
double_pendulum/
  config.py              plant constants + DR + hyperparam dicts
  model.xml              MJCF (Pinocchio-compatible)
lib/
  networks.py            flax MLPs: pure controller / controller / theta estimator
  domain_randomization.py  sample_theta + apply_theta
  rollout.py             unified MJX closed-loop rollout
  losses.py              tracking / control / theta / endpoint losses (angle-wrapped)
train/
  solve_trajectory.py    Crocoddyl FDDP on nominal plant, N trajectories
  train_pure.py          BPTT through MJX, pure-MLP residual baseline
  train_theta_estimator.py   PD-only rollouts → sysid MLP (no BPTT)
  train_controller.py    BPTT through MJX, two-model with frozen θ̂
eval/
  evaluate.py            1000 held-out rollouts × 3 controllers, endpoint + tracking
  evaluate_pure.py       pd vs pure only (skips two-model), adds |v|rms
  evaluate_estimator.py  per-parameter θ identifiability (R² / RMSE)
  plot_final.py          loss curves + eval-metric figures
  plot_trajectories.py   TO reference sanity check (angles + cartesian strobe)
  plot_eval_rollouts.py  closed-loop rollout viz, pd vs pure
outputs/double_pendulum/ artifacts written by the pipeline
```

## Dependencies

Linux + NVIDIA GPU. Crocoddyl + Pinocchio + JAX + MJX are Linux-only via conda-forge.
The code does not run on Windows (some pip installs hit Windows MAX_PATH limits and
Crocoddyl is Linux-only on conda-forge).
