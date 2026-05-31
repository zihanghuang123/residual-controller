"""Medium seven-pendulum config: 256x256, w=200, H=600.

One step up from config.py along both axes -- capacity + horizon doubled.
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
NQ = 7
NV = 7
NU = 7
N_LINKS = 7


# Trajectory horizon
TIMESTEP = 0.002
SIM_DURATION = 5.0
N_STEPS = int(SIM_DURATION / TIMESTEP)


# PD gains (per-joint). Tapered with depth: each level is 1/4 of the one above.
KP = np.array([120.0, 30.0, 7.5, 1.88, 0.47, 0.12, 0.03])
KD = np.array([6.0, 1.5, 0.38, 0.09, 0.02, 0.006, 0.0015])


# Library of (x0, xf) pairs. Root joint swings up to pi; the rest target 0.
N_TRAJECTORIES = 200
INITIAL_QPOS_RANGE = (np.full(N_LINKS, -0.5), np.full(N_LINKS, 0.5))
_TARGET_LO = np.full(N_LINKS, -0.5)
_TARGET_HI = np.full(N_LINKS, 0.5)
_TARGET_LO[0] = np.pi - 0.5
_TARGET_HI[0] = np.pi + 0.5
TARGET_QPOS_RANGE = (_TARGET_LO, _TARGET_HI)
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
    "hidden_sizes": (256, 256),
    "n_history": 200,
    "n_rollout": 600,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 3000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Theta estimator
THETA = {
    "hidden_sizes": (512, 512),
    "n_history": 200,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 3000,
    "grad_clip_norm": 1.0,
}


# Controller with frozen theta estimator
CONTROLLER = {
    "hidden_sizes": (256, 256),
    "n_history": 200,
    "n_rollout": 600,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 3000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Oracle controller (upper bound for two-model)
ORACLE = {
    "hidden_sizes": (256, 256),
    "n_history": 200,
    "n_rollout": 600,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 3000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Evaluation
N_EVAL_PLANTS = 200
EVAL_SEED = 42
