"""Plant-specific config for the Kinova Gen3 (7-DoF arm).

PD gains, target ranges, and TO cost weights are starting points -- validate against
solve_trajectory.py + plot_trajectories.py before training (gravity, no joint limits in model).
"""

from pathlib import Path

import numpy as np


# Paths
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
MODEL_PATH = HERE / "kinova_gen3.xml"
PLANT_NAME = HERE.name
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem


# Plant dimensions
NQ = 7
NV = 7
NU = 7
N_LINKS = 7


# Trajectory horizon
TIMESTEP = 0.002
SIM_DURATION = 3.0
N_STEPS = int(SIM_DURATION / TIMESTEP)


# PD gains (per-joint). Scaled to joint torque limits (39 Nm proximal, 9 Nm distal). UNTUNED.
KP = np.array([200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0])
KD = np.array([20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0])


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
    "inertia_scale": (0.7, 1.3),
}
THETA_DIM = 4 * N_LINKS


# Pure MLP residual
PURE = {
    "hidden_sizes": (512, 512),
    "n_history": 300,
    "n_rollout": 500,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 8000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Pure RNN residual (BPTT). n_rollout to be set from estimate_horizon.py.
PURE_RNN = {
    "hidden_sizes": (256, 256),
    "n_rollout": 500,
    "batch_size": 128,
    "lr": 3e-4,
    "n_iterations": 20000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-9,
}


# Evaluation
N_EVAL_PLANTS = 2000
EVAL_SEED = 42
