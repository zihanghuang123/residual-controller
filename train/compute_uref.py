"""Fill u_ref into trajectories.npz via MJX inverse dynamics (run in the training env).

Reads the states (x_refs, qacc_refs) that solve_trajectory_phlame.py wrote and adds u_refs
= mjx.inverse(model, q, qd, qdd) on the nominal model -- full dynamics (armature, damping,
gravity, Coriolis). Run after solve_trajectory_phlame.py, before plot_trajectories.py / training.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from lib import training


def main() -> None:
    cfg = training.load_config()
    N = cfg.NQ
    traj_path = cfg.OUTPUT_DIR / "trajectories.npz"

    data = dict(np.load(traj_path))
    if "qacc_refs" not in data:
        raise SystemExit(f"{traj_path} has no qacc_refs -- run solve_trajectory_phlame.py first")
    x_refs = data["x_refs"]            # (M, T+1, 2N)
    qacc = data["qacc_refs"]           # (M, T+1, N)
    M, T_plus_1, _ = x_refs.shape

    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_nominal = mjx.put_model(mj_model)
    inverse_over_time = jax.jit(jax.vmap(training.inverse_dynamics, in_axes=(None, 0, 0, 0)))

    print(f"computing u_ref for {M} trajectories x {T_plus_1} steps")
    u_refs = np.zeros((M, T_plus_1 - 1, N))
    for i in range(M):
        u = np.asarray(inverse_over_time(
            mjx_nominal,
            jnp.asarray(x_refs[i, :, :N]),
            jnp.asarray(x_refs[i, :, N:]),
            jnp.asarray(qacc[i]),
        ))
        u_refs[i] = u[:-1]

    data["u_refs"] = u_refs
    np.savez(traj_path, **data)
    conv = data["converged"].astype(bool)
    print(f"wrote u_refs {u_refs.shape} into {traj_path}")
    print(f"max |u_ref| over converged: {np.abs(u_refs[conv]).max():.1f}")


if __name__ == "__main__":
    main()
