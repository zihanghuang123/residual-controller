# Neural Residual Controller

An implementation of a neural residual controller robust to plant/model mismatch. Trained end-to-end via backpropagation through differentiable physics (MJX) under domain randomization.

## Formulation

The control law applied to the plant at every step:

```
u(t) = u*(t) + PD(x*(t) - x(t)) + v(t)
```

where `u*` and `x*` come from offline trajectory optimization on the _nominal_ plant, `PD` is a fixed feedback term, and `v(t)` is a learned residual whose purpose is to correct for the gap between the nominal plant (used by TO) and the actual plant.

Four options for `v(t)`:

- **PD-only**: `v(t) = 0` (no residual)
- **Pure MLP**: `v(t) = MLP(history, reference)`
- **Two-model**: `v(t) = MLP(history, reference, θ_hat)` where `θ_hat = estimator(history)`
- **Oracle**: `v(t) = MLP(history, reference, θ)` — upper bound for two-model, fed the true θ

Pure MLP is the natural baseline ("give the network everything and let it figure out the plant"). Two-model decouples the problem: a separate frozen estimator does system identification from observed `(x, u)` history, and the controller gets `θ_hat` as an explicit input. Oracle is what two-model could achieve if the estimator were perfect.

## Domain randomization

Each rollout samples a fresh plant from `cfg.DR_RANGES`:

- `mass_scale` per link — multiplies nominal body mass
- `damping` per link — applied to each DoF
- `frictionloss` per link — Coulomb friction on each DoF

Theta is applied to the MJX model in-place via `apply_theta` before each rollout.

The estimator's input is **history only** — `(x_hist, u_hist)`. References carry no information about theta and are excluded.

## Truncated BPTT

Trajectories from `solve_trajectory.py` have horizon `T = SIM_DURATION / TIMESTEP` steps (1000 for the double pendulum at 2 s, 2 ms). Backpropagating through all `T` MJX steps per iteration is impractical: memory grows linearly with `T`, and gradient norms can explode through long unrolls of stiff dynamics.

Instead, each training iteration samples a random window of length `H = cfg.PURE["n_rollout"]` from a random TO trajectory and a random start index `t0`. The rollout starts at the reference state `x_refs[idx, t0]`, runs for `H` MJX steps under the closed-loop law, and the loss is computed over the resulting `H+1` states:

```
tracking_loss(xs_full, x_refs[idx, t0:t0+H+1], nq) + alpha_reg * reg_loss(vs)
```

Each iteration sees a different combination of (trajectory, start time, plant). The history buffers are initialized by padding with `x_refs[idx, t0]` — matching what deployment will see in its first few steps.

At evaluation, rollouts run the **full `T` steps** to test whether each controller actually completes the task end-to-end. This train-vs-eval horizon mismatch is real and can show up as a long-horizon generalization gap (more training → larger `|v|` → compounding errors over `T > H` steps).

### Choosing the BPTT horizon

`H` has an **optimum** — longer is not better. For chaotic plants the BPTT gradient variance grows like `exp(2·λ·H)` (`λ` = largest closed-loop Lyapunov exponent), while the bias from truncated credit assignment stops falling once the controller stabilizes. Past a plant-dependent horizon the gradient is noise-dominated and training oscillates instead of converging (empirically the six-link GRU trains well at `H ≈ 400–500` but diverges by `H ≈ 1000`). Two scripts predict the usable range without a full sweep:

- `eval/estimate_horizon.py` — measures `λ` by jvp power iteration through the PD-closed-loop dynamics; reports the Lyapunov time `τ = 1/λ` and a horizon band. Pre-training (needs no learned params).

## Pipeline

Core run order — `train/` produces artifacts, `eval/` consumes them. The `--config` flag selects the plant config; default is `plants/double_pendulum/config.py`.

```
python train/solve_trajectory.py         # → trajectories.npz
python train/train_pure.py               # → pure_params.pkl
python train/train_oracle.py             # → oracle_params.pkl
python train/train_theta_estimator.py    # → theta_params.pkl
python train/train_controller.py         # → controller_params.pkl
python eval/evaluate_pure.py             # → metrics_pure.npz + eval plots (pd vs pure)
```

For a different plant, point at its config:

```
python train/solve_trajectory.py --config plants/triple_pendulum/config.py
python train/train_pure.py --config plants/triple_pendulum/config.py
# ... etc
```

Additional eval / diagnostic scripts:

```
python eval/plot_trajectories.py         # TO reference sanity check (no MuJoCo)
python eval/plot_residual_check.py       # u_nom / +pd / computed-torque expert sanity check
python eval/evaluate_oracle.py           # pd vs oracle only
python eval/evaluate_estimator.py        # per-parameter θ R² / RMSE
python eval/estimate_horizon.py          # Lyapunov-time BPTT horizon estimate (pre-training)
python eval/grad_noise_horizon.py        # gradient-noise-scale horizon crossover
python eval/plot_h_sweep.py              # endpoint/tracking box plots across an H sweep
```

RNN variants of the controller and estimator:

```
python train/train_pure_rnn.py
python train/train_theta_estimator_rnn.py
python eval/evaluate_pure_rnn.py          # pd vs pure_rnn: metrics + endpoint/tracking box + per-traj tracking
```

Outputs land in `cfg.OUTPUT_DIR`, which defaults to `outputs/<plant>/<config_stem>/`. Different config files on the same plant (e.g. `config.py` vs `config_big.py`) write to different output directories and can be trained in parallel on the same GPU without collision.

### Training queue

Per-host queue scripts run jobs **sequentially per GPU, GPUs in parallel**: `train_queue_bloodseeker.sh` / `train_queue_bane.sh` (2 GPUs, `GPU0`/`GPU1` arrays) and `train_queue_blob.sh` (1 GPU, `GPU0`). Each array entry is `"<command> | <log name>"` and logs to `logs/<name>.log`; an empty array leaves that GPU idle. To choose what runs, edit only the `GPU0`/`GPU1` arrays.

```
tmux new -s train
bash train_queue_bloodseeker.sh    # Ctrl+B then D to detach
```

Entries run in listed order, so a multi-step job is written as consecutive entries (e.g. `mkdir` / `cp trajectories` / `train`). The arrays currently run the four/six-link BPTT horizon sweep over the `plants/<plant>/hNNN.py` configs.

## Porting to a different plant

Plant-specific code lives in `plants/<plant>/` directories. To port to a new robot:

1. **Create `plants/<plant>/model.xml`** (MJCF). Must be Pinocchio-compatible: avoid features `buildModelFromMJCF` rejects (mesh decompositions, exotic actuator types). Set `contype=0 conaffinity=0` to disable contact if contact-free training is desired.

2. **Create `plants/<plant>/config.py`** following the structure of `plants/double_pendulum/config.py`. At minimum:
   - `OUTPUT_DIR`, `MODEL_PATH`, `PLANT_NAME` (all derived from `__file__`)
   - `NQ`, `NV`, `NU`, `N_LINKS`
   - `TIMESTEP`, `SIM_DURATION` (often needs retuning — fast plants need smaller dt)
   - `KP`, `KD` — per-DoF feedback gains (usually tapered with joint depth on multi-link arms)
   - `INITIAL_QPOS_RANGE`, `TARGET_QPOS_RANGE`, `N_TRAJECTORIES`
   - `TO_COST_X_RUNNING`, `TO_COST_U_RUNNING`, `TO_COST_X_TERMINAL` — TO weights; often need retuning per plant
   - `DR_RANGES` and `THETA_DIM` — if the DR structure changes (different per-link parameters), also edit `lib/domain_randomization.py`
   - `PURE`, `THETA`, `CONTROLLER`, `ORACLE` hyperparameter dicts (RNN variants if used)

3. **Run with `--config plants/<plant>/config.py`** — no source edits needed.

## Repo layout

```
plants/
  double_pendulum/
    config.py            plant constants + DR + hyperparam dicts
    model.xml            MJCF (Pinocchio-compatible)
  triple_pendulum/       config.py (+ config1/config2 size variants), model.xml
  four_pendulum/ ... seven_pendulum/   same structure, more links
  kinova/                Kinova Gen3 assets (WIP)
lib/
  networks.py            flax MLPs + GRUs: pure controller / controller / θ estimator
  domain_randomization.py  sample_theta + apply_theta
  rollout.py             MJX closed-loop rollout (MLP-style history-buffer + RNN-style hidden-state)
  losses.py              tracking / control / theta / endpoint losses (angle-wrapped)
  training.py            shared scaffolding: --config arg, MJX setup, BPTT train loop
  evaluation.py          shared scaffolding: vmap over plants, metric summary, npz save
train/
  solve_trajectory.py        Crocoddyl FDDP on nominal plant, N trajectories
  train_pure.py              BPTT through MJX, pure-MLP residual
  train_oracle.py            same arch as two-model, fed ground-truth θ (upper bound)
  train_theta_estimator.py   PD-only rollouts → sysid MLP (no BPTT)
  train_controller.py        BPTT through MJX, two-model with frozen θ̂
  train_pure_rnn.py          GRU variant of pure
  train_theta_estimator_rnn.py  GRU variant of θ estimator
eval/
  evaluate_pure.py       pd vs pure: metrics + endpoint/tracking box + per-traj tracking plots
  evaluate_oracle.py     pd vs oracle only
  evaluate_pure_rnn.py   pd vs pure_rnn: same metrics + plots as evaluate_pure
  evaluate_estimator.py  per-parameter θ R² / RMSE
  estimate_horizon.py    Lyapunov-time BPTT horizon estimate (jvp power iteration)
  grad_noise_horizon.py  gradient-noise-scale horizon crossover vs batch size
  plot_h_sweep.py        endpoint/tracking box plots across an H sweep
  plot_trajectories.py   TO reference sanity check (angles + cartesian strobe)
  plot_residual_check.py u_nom / +pd / computed-torque expert sanity check
train_queue_*.sh         per-host GPU-array queues (bloodseeker/bane: 2 GPU, blob: 1 GPU)
outputs/<plant>/<config_stem>/  artifacts written by the pipeline
```

## Dependencies

Linux + NVIDIA GPU. Crocoddyl + Pinocchio + JAX + MJX are Linux-only via conda-forge. On Windows, run inside WSL2 (Ubuntu) — that's the tested path. On shared compute where `sudo apt install` isn't allowed, build inside a Docker container — NVIDIA's `nvcr.io/nvidia/jax:24.04-py3` image is a good base, with `--gpus all -v $PWD:/workspace` for code mounting. See `environment.yml` for the conda recipe.
