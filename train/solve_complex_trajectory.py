"""Chain N single-target TOs into one long swing-up/swing-down trajectory.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lib import training
from train.solve_trajectory import build_model, solve_one

# How many alternating swing-up/swing-down segments per complex trajectory.
N_CHAIN_SEGMENTS = 6


def sample_chain_targets(cfg, n_chain_segments: int, seed: int):
    """For one complex trajectory: an initial state at the bottom + a list of alternating targets.
    """
    rng = np.random.default_rng(seed)
    qpos_init_lo, qpos_init_hi = cfg.INITIAL_QPOS_RANGE
    qpos_target_lo, qpos_target_hi = cfg.TARGET_QPOS_RANGE

    qpos_init = rng.uniform(qpos_init_lo, qpos_init_hi, size=cfg.NQ)
    x_init = np.hstack([qpos_init, np.zeros(cfg.NQ)])

    x_targets = []
    for k in range(n_chain_segments):
        if k % 2 == 0:
            qpos = rng.uniform(qpos_target_lo, qpos_target_hi, size=cfg.NQ)   # up
        else:
            qpos = rng.uniform(qpos_init_lo, qpos_init_hi, size=cfg.NQ)       # down
        x_targets.append(np.hstack([qpos, np.zeros(cfg.NQ)]))

    return x_init, x_targets


def solve_chain(cfg, model, x_init, x_targets):
    """Solve a sequence of TOs, chaining segment k's final state into segment k+1's initial state.
    """
    all_xs = []
    all_us = []
    converged_all = True

    current_x = x_init
    last_xs_tail = None
    for seg_idx, target in enumerate(x_targets):
        xs, us, conv = solve_one(cfg, model, current_x, target)
        # Drop the segment's last state to avoid duplicating it with the next segment's first.
        all_xs.append(xs[:-1])
        all_us.append(us)
        converged_all = converged_all and bool(conv)
        current_x = np.asarray(xs[-1])
        last_xs_tail = xs[-1:]

    # Append the very last state of the chain.
    if last_xs_tail is not None:
        all_xs.append(last_xs_tail)

    xs_concat = np.concatenate(all_xs, axis=0)   # (n_segments * N_STEPS + 1, 2*nq)
    us_concat = np.concatenate(all_us, axis=0)   # (n_segments * N_STEPS, nu)
    return xs_concat, us_concat, converged_all


def main() -> None:
    cfg = training.load_config()
    output_path = cfg.OUTPUT_DIR / "complex_trajectories.npz"

    print(f"loading Pinocchio model from {cfg.MODEL_PATH}")
    model = build_model(cfg)
    print(f"  nq={model.nq}  nv={model.nv}")

    n_seg = N_CHAIN_SEGMENTS
    total_T = n_seg * cfg.N_STEPS
    print(f"sampling {cfg.N_TRAJECTORIES} complex trajectories")
    print(f"  {n_seg} segments each (alternating up/down), total T={total_T} steps")

    # Per-chain seed sequence so each chain is independent but reproducible.
    rng_master = np.random.default_rng(cfg.TRAJECTORY_SAMPLE_SEED)

    x_refs = []
    u_refs = []
    x_inits = []
    x_targets_all = []   # (N, n_seg, 2*nq)
    converged_flags = []

    for i in range(cfg.N_TRAJECTORIES):
        seed_i = int(rng_master.integers(0, 2**31 - 1))
        x_init, targets = sample_chain_targets(cfg, n_seg, seed_i)

        xs, us, conv = solve_chain(cfg, model, x_init, targets)

        x_refs.append(xs)
        u_refs.append(us)
        x_inits.append(x_init)
        x_targets_all.append(np.stack(targets, axis=0))
        converged_flags.append(conv)

        status = "OK  " if conv else "FAIL"
        print(f"  chain {i + 1:3d}/{cfg.N_TRAJECTORIES}  {status}  ({n_seg} segments)")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x_refs=np.stack(x_refs),               # (N, n_seg * N_STEPS + 1, 2*nq)
        u_refs=np.stack(u_refs),               # (N, n_seg * N_STEPS,    nu)
        x_inits=np.stack(x_inits),             # (N, 2*nq)
        x_targets=np.stack(x_targets_all),     # (N, n_seg, 2*nq)
        converged=np.array(converged_flags),   # (N,)
        n_chain_segments=n_seg,
    )
    n_ok = sum(converged_flags)
    print(f"saved {output_path}")
    print(f"converged (all {n_seg} segments): {n_ok}/{cfg.N_TRAJECTORIES}")


if __name__ == "__main__":
    main()
