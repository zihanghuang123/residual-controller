"""Plant-specific config for the double pendulum."""

from pathlib import Path

import numpy as np


# Paths
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
MODEL_PATH = HERE / "model.xml"
PLANT_NAME = HERE.name
OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem


# Plant dimensions
NQ = 2
NV = 2
NU = 2
N_LINKS = 2


# Trajectory horizon
TIMESTEP = 0.002 
SIM_DURATION = 2.0
N_STEPS = int(SIM_DURATION / TIMESTEP)


# PD gains (per-joint, applied in closed-loop residual rollout)
KP = np.array([20.0, 5.0])
KD = np.array([1.0, 0.2])


# Library of (x0, xf) pairs: set N_TRAJECTORIES = 1 for single-trajectory baseline.
N_TRAJECTORIES = 50
INITIAL_QPOS_RANGE = (np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
TARGET_QPOS_RANGE = (np.array([np.pi - 0.5, -0.5]), np.array([np.pi + 0.5, 0.5]))
TRAJECTORY_SAMPLE_SEED = 42


# Trajectory optimization (Crocoddyl FDDP)
# J = sum_k [w_x ||x_k - x_target||^2 + w_u ||u_k||^2] + w_xT ||x_N - x_target||^2
TO_COST_X_RUNNING = 1.0
TO_COST_U_RUNNING = 1e-3
TO_COST_X_TERMINAL = 100.0


# Domain randomization
# Each rollout samples a fresh theta from this distribution.
DR_RANGES = {
    "mass_scale": (0.7, 1.3), 
    "damping": (0.0, 0.5), 
    "frictionloss": (0.0, 0.2)
}
THETA_DIM = 3 * N_LINKS 


# Pure MLP residual
# Input:  history (x, u) + reference (x*, u*)
# Output: v in R^NU
PURE = {
    "hidden_sizes": (128, 128),
    "n_history": 100,        # w
    "n_rollout": 300,       # H (BPTT window)
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 6000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Theta estimator
# Input:  history (x, u) + reference (x*, u*)
# Output: theta_estimate in R^THETA_DIM
THETA = {
    "hidden_sizes": (512, 512),
    "n_history": 100,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 10000,
    "grad_clip_norm": 1.0,
}


# Controller with frozen theta estimator
# Input:  history (x, u) + reference (x*, u*) + theta_estimate
# Output: v in R^NU
CONTROLLER = {
    "hidden_sizes": (128, 128),
    "n_history": 100,
    "n_rollout": 300,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 6000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Oracle controller (upper bound for two-model)
# Input:  history (x, u) + reference (x*, u*) + true_theta
# Output: v in R^NU
ORACLE = {
    "hidden_sizes": (128, 128),
    "n_history": 100,
    "n_rollout": 300,
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 6000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Pure RNN residual
# Per-step input:  (x_t, u_{t-1}, x_ref_t, u_ref_t)
# Output:          v in R^NU
PURE_RNN = {
    "hidden_sizes": (128, 128),  
    "n_rollout": 300,             
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 6000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Theta estimator (streaming RNN)
# Per-step input:  (x_t, u_{t-1})
# Output:          theta_estimate in R^THETA_DIM
THETA_RNN = {
    "hidden_sizes": (128, 128),  
    "n_rollout": 300,     
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 6000,
    "grad_clip_norm": 1.0,
}


# Controller RNN with frozen theta estimator
# Per-step input:  (x_t, u_{t-1}, x_ref_t, u_ref_t, theta_hat)
# Output:          v in R^NU
CONTROLLER_RNN = {
    "hidden_sizes": (128, 128),  
    "n_rollout": 300,         
    "batch_size": 64,
    "lr": 3e-4,
    "n_iterations": 3000,
    "grad_clip_norm": 1.0,
    "alpha_reg": 1e-5,
}


# Evaluation
N_EVAL_PLANTS = 1000
EVAL_SEED = 42
