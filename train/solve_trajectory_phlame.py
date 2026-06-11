"""AGHF (PHLAME) trajectory solver -- states only. Run compute_uref.py next for u_ref."""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from phlame.aghf import PostAghf
from phlame.experiment import Experiment
from phlame.parameter_set import ParameterSetBase

from lib.sampling import sample_targets


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


def load_config():
    """Local (no lib.training): keeps this solver free of the jax/mujoco training stack."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("plant_cfg", Path(args.config).resolve())
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


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
    cfg = load_config()
    N = cfg.NQ
    T = cfg.SIM_DURATION
    t_scale = T / 2.0
    fp_urdf = str(cfg.PHLAME_URDF)

    t_interp = np.linspace(-1.0, 1.0, cfg.N_STEPS + 1)
    x_inits, x_targets = sample_targets(cfg, cfg.N_TRAJECTORIES, cfg.TRAJECTORY_SAMPLE_SEED)

    x_refs = np.zeros((cfg.N_TRAJECTORIES, cfg.N_STEPS + 1, 2 * N))
    qacc_refs = np.zeros((cfg.N_TRAJECTORIES, cfg.N_STEPS + 1, N))
    converged = np.zeros(cfg.N_TRAJECTORIES, dtype=bool)
    solve_times = np.zeros(cfg.N_TRAJECTORIES)

    print(f"AGHF: {cfg.N_TRAJECTORIES} trajectories, N={N}, T={T}s, dt={cfg.TIMESTEP}")

    for i in range(cfg.N_TRAJECTORIES):
        try:
            q, qd, qdd, t_solve = solve_one(
                N, t_scale, x_inits[i], x_targets[i], fp_urdf, t_interp,
                run_name=f"{cfg.PLANT_NAME}_{i:04d}",
            )
            x_refs[i, :, :N] = q
            x_refs[i, :, N:] = qd
            qacc_refs[i] = qdd
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
        qacc_refs=qacc_refs,
        x_inits=x_inits,
        x_targets=x_targets,
        converged=converged,
        solve_times=solve_times,
    )
    n_ok = int(converged.sum())
    avg_t = float(solve_times[converged].mean()) if n_ok else 0.0
    print(f"\nsaved {output_path} (states only; no u_ref yet)")
    print(f"converged {n_ok}/{cfg.N_TRAJECTORIES}, avg solve {avg_t:.2f}s")
    print("next: `python train/compute_uref.py --config <cfg>` in the training env")


if __name__ == "__main__":
    main()
