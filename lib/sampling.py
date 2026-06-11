"""Plant boundary-condition sampling for trajectory optimization (numpy-only, env-light)."""

import numpy as np


def sample_targets(cfg, n_trajectories: int, seed: int):
    """Sample N (x_init, x_target) pairs uniformly from config qpos/qvel ranges (shared seed -> shared BCs)."""
    rng = np.random.default_rng(seed)
    qpos_init_lo, qpos_init_hi = cfg.INITIAL_QPOS_RANGE
    qpos_target_lo, qpos_target_hi = cfg.TARGET_QPOS_RANGE
    qvel_init_lo, qvel_init_hi = cfg.INITIAL_QVEL_RANGE
    qvel_target_lo, qvel_target_hi = cfg.TARGET_QVEL_RANGE

    qpos_inits = rng.uniform(qpos_init_lo, qpos_init_hi, size=(n_trajectories, cfg.NQ))
    qpos_targets = rng.uniform(qpos_target_lo, qpos_target_hi, size=(n_trajectories, cfg.NQ))
    qvel_inits = rng.uniform(qvel_init_lo, qvel_init_hi, size=(n_trajectories, cfg.NV))
    qvel_targets = rng.uniform(qvel_target_lo, qvel_target_hi, size=(n_trajectories, cfg.NV))

    return (np.hstack([qpos_inits, qvel_inits]),
            np.hstack([qpos_targets, qvel_targets]))
