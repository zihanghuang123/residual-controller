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
PHLAME_URDF = HERE / "kinova.urdf"
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


KP = np.array([80.0, 80.0, 80.0, 80.0, 18.0, 18.0, 18.0])  
KD = np.array([35.0, 35.0, 35.0, 35.0, 12.0, 12.0, 12.0]) 


# kinova.urdf limits (MJCF ctrlrange matches U_MAX)
U_MAX = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
TO_U_MAX = 0.85 * U_MAX
QPOS_LIMIT = np.array([6.28, 2.25, 6.28, 2.58, 6.28, 2.1, 6.28])
QVEL_LIMIT = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
_SAMPLE_QPOS = np.minimum(QPOS_LIMIT, np.pi)

N_TRAJECTORIES = 2000
INITIAL_QPOS_RANGE = (-_SAMPLE_QPOS, _SAMPLE_QPOS)
TARGET_QPOS_RANGE = (-_SAMPLE_QPOS, _SAMPLE_QPOS)
INITIAL_QVEL_RANGE = (-QVEL_LIMIT, QVEL_LIMIT)
TARGET_QVEL_RANGE = (-QVEL_LIMIT, QVEL_LIMIT)
TRAJECTORY_SAMPLE_SEED = 42


# Trajectory optimization (Crocoddyl FDDP)
TO_COST_X_RUNNING = 1.0
TO_COST_U_RUNNING = 1e-3
TO_COST_X_TERMINAL = 100.0


# Domain randomization
DR_RANGES = {
    "mass_scale": (0.7, 1.3),
    "damping": (-0.25, 0.25),
    "frictionloss": (-0.1, 0.1),
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
N_EVAL_PLANTS = 200
EVAL_SEED = 42
