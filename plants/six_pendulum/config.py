"""Plant-specific config for the six-link pendulum.

Same structure as double_/triple_pendulum, NQ/NV/NU=6. KP/KD taper with joint
depth (factor 4 per level, matching the 2-/3-link convention). The deep-joint
gains and SIM_DURATION are extrapolated and likely need retuning -- validate the
references with solve_trajectory.py + plot_trajectories.py before training.
"""

from pathlib import Path

import numpy as np


# Paths
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent  # repo root (plants/<plant>/)
MODEL_PATH = HERE / "model.xml"
PLANT_NAME = HERE.name
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem


# Plant dimensions
NQ = 6
NV = 6
NU = 6
N_LINKS = 6


# Trajectory horizon
TIMESTEP = 0.002
SIM_DURATION = 2.0
N_STEPS = int(SIM_DURATION / TIMESTEP)


# PD gains (per-joint). Tapered with depth: each level is 1/4 of the one above.
KP = np.array([100.0, 25.0, 6.25, 1.56, 0.39, 0.1])
KD = np.array([5.0, 1.25, 0.31, 0.08, 0.02, 0.005])


N_TRAJECTORIES = 2000
INITIAL_QPOS_RANGE = (np.full(N_LINKS, -np.pi), np.full(N_LINKS, np.pi))
TARGET_QPOS_RANGE = (np.full(N_LINKS, -np.pi), np.full(N_LINKS, np.pi))
INITIAL_QVEL_RANGE = (np.full(N_LINKS, -3.0), np.full(N_LINKS, 3.0))
TARGET_QVEL_RANGE = (np.full(N_LINKS, -3.0), np.full(N_LINKS, 3.0))
TRAJECTORY_SAMPLE_SEED = 42


# Trajectory optimization (Crocoddyl FDDP)
TO_COST_X_RUNNING = 1.0
TO_COST_U_RUNNING = 1e-3
TO_COST_X_TERMINAL = 100.0


# Domain randomization
DR_RANGES = {
    "mass_scale": (0.7, 1.3),
    "damping": (0.0, 0.5),
    "frictionloss": (0.0, 0.2),
}
THETA_DIM = 3 * N_LINKS


# Pure MLP residual
PURE = {
    "hidden_sizes": (512, 512),
    "n_history": 300,
    "n_rollout": 1500,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 8000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Theta estimator
THETA = {
    "hidden_sizes": (512, 512),
    "n_history": 300,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 8000,
    "grad_clip_norm": 1.0,
}


# Controller with frozen theta estimator
CONTROLLER = {
    "hidden_sizes": (512, 512),
    "n_history": 300,
    "n_rollout": 1500,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 8000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Oracle controller (upper bound for two-model)
ORACLE = {
    "hidden_sizes": (512, 512),
    "n_history": 300,
    "n_rollout": 1500,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 8000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Evaluation
N_EVAL_PLANTS = 200
EVAL_SEED = 42
