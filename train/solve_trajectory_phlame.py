"""AGHF (PHLAME) trajectory solver. Drop-in for solve_trajectory.py."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pinocchio as pin

from phlame.aghf import PostAghf
from phlame.control_extraction import compute_u_matrix_static
from phlame.experiment import Experiment
from phlame.parameter_set import ParameterSetBase

from lib import training


# AGHF settings (from PHLAME/examples/kinova.py)
P_NODES = 7
K_PENALTY = 1e4
S_MAX = 1
ABS_TOL = 1e-4
REL_TOL = 1e-4
METHOD_NAME = "cvode"
MAX_STEPS = int(1e8)
NS_POINTS = int(1e2)
TIMEOUT_SEC = 200
USE_JACOBIAN = True


def sample_targets(cfg, n_trajectories: int, seed: int):
    """Same sampling as solve_trajectory.py (shared seed -> shared BCs)."""
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


def mjcf_dof_dynamics(model_path: str, n: int):
    """Per-DOF (armature, damping, frictionloss) from the MJCF named joints."""
    joints = [j for j in ET.parse(model_path).getroot().iter("joint") if j.get("name")]
    assert len(joints) == n, f"{len(joints)} named joints, expected {n}"
    armature = np.array([float(j.get("armature", 0.0)) for j in joints])
    damping = np.array([float(j.get("damping", 0.0)) for j in joints])
    frictionloss = np.array([float(j.get("frictionloss", 0.0)) for j in joints])
    return armature, damping, frictionloss


def solve_one(N, t_scale, x_init, x_target, fp_urdf, t_interp, run_name):
    """Solve one AGHF problem. Returns (q, qd_phys, qdd_phys, t_solve)."""
    j_type = (2 * np.ones((N, 1))).astype(np.double, order="F")

    # velocity BCs: physical -> normalized time
    X0 = x_init.astype(np.double).copy(); X0[N:] *= t_scale
    Xf = x_target.astype(np.double).copy(); Xf[N:] *= t_scale
    X0 = X0.reshape(-1, 1).copy(order="F")
    Xf = Xf.reshape(-1, 1).copy(order="F")

    pset = ParameterSetBase(
        p=P_NODES, N=N, X0=X0, Xf=Xf, name=run_name, s_max=S_MAX, k=K_PENALTY,
        abs_tol=ABS_TOL, rel_tol=REL_TOL, method_name=METHOD_NAME, max_steps=MAX_STEPS,
        ns_points=NS_POINTS, j_type=j_type, fp_urdf=fp_urdf,
    )
    result = Experiment.run_single_static(
        pset=pset, timeout=TIMEOUT_SEC, use_jacobian=USE_JACOBIAN, print_debug=False,
    )

    post = PostAghf(result)
    q, qd, qdd = post.get_q_qd_qdd(result.sol[-1, :], t_interp)
    return q, qd / t_scale, qdd / (t_scale ** 2), result.t_solve


def main() -> None:
    cfg = training.load_config()
    N = cfg.NQ
    T = cfg.SIM_DURATION
    t_scale = T / 2.0
    fp_urdf = str(cfg.PHLAME_URDF)

    pin_model = pin.buildModelFromUrdf(fp_urdf)  # rigid body == MJCF
    armature, damping, frictionloss = mjcf_dof_dynamics(str(cfg.MODEL_PATH), N)

    t_interp = np.linspace(-1.0, 1.0, cfg.N_STEPS + 1)
    x_inits, x_targets = sample_targets(cfg, cfg.N_TRAJECTORIES, cfg.TRAJECTORY_SAMPLE_SEED)

    x_refs = np.zeros((cfg.N_TRAJECTORIES, cfg.N_STEPS + 1, 2 * N))
    u_refs = np.zeros((cfg.N_TRAJECTORIES, cfg.N_STEPS, N))
    converged = np.zeros(cfg.N_TRAJECTORIES, dtype=bool)
    solve_times = np.zeros(cfg.N_TRAJECTORIES)

    print(f"AGHF: {cfg.N_TRAJECTORIES} trajectories, N={N}, T={T}s, dt={cfg.TIMESTEP}")

    for i in range(cfg.N_TRAJECTORIES):
        try:
            q, qd, qdd, t_solve = solve_one(
                N, t_scale, x_inits[i], x_targets[i], fp_urdf, t_interp,
                run_name=f"{cfg.PLANT_NAME}_{i:04d}",
            )
            # full inverse dynamics: rigid-body rnea + armature + damping + Coulomb friction
            u = compute_u_matrix_static(q, qd, qdd, pin_model)
            u = u + qdd * armature + qd * damping + np.sign(qd) * frictionloss
            x_refs[i, :, :N] = q
            x_refs[i, :, N:] = qd
            u_refs[i] = u[:-1]
            converged[i] = True
            solve_times[i] = t_solve
            print(f"  {i + 1:4d}/{cfg.N_TRAJECTORIES}  OK  ({t_solve:.2f}s)")
        except Exception as e:
            print(f"  {i + 1:4d}/{cfg.N_TRAJECTORIES}  FAIL: {type(e).__name__}: {e}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = cfg.OUTPUT_DIR / "trajectories.npz"
    np.savez(
        output_path,
        x_refs=x_refs,
        u_refs=u_refs,
        x_inits=x_inits,
        x_targets=x_targets,
        converged=converged,
        solve_times=solve_times,
    )
    n_ok = int(converged.sum())
    avg_t = float(solve_times[converged].mean()) if n_ok else 0.0
    print(f"\nsaved {output_path}")
    print(f"converged {n_ok}/{cfg.N_TRAJECTORIES}, avg solve {avg_t:.2f}s")


if __name__ == "__main__":
    main()
