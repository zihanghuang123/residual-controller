"""PHLAME-based replacement for solve_trajectory.py. Generates an MJCF-matched URDF on the fly and writes trajectories.npz in the same format."""

import argparse
import importlib.util
import os
import sys
import tempfile
from types import ModuleType

import numpy as np

from phlame.aghf import PostAghf
from phlame.parameter_set import ParameterSetBase
from phlame.experiment import Experiment


LINK_LENGTH = 1.0
LINK_MASS = 1.0
COM_OFFSET = 1.0

PHLAME_T_SIM = 2.0

P_NODES = 7
K_PENALTY = 1e3
S_MAX = 10
ABS_TOL = 1e-10
REL_TOL = 1e-10
METHOD_NAME = "cvode"
MAX_STEPS = 1e2
NS_POINTS = int(1e2)
TIMEOUT_SEC = 60


def load_config(config_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("plant_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_targets(cfg, n_trajectories: int, seed: int):
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


def generate_matched_urdf(n_links: int, name: str) -> str:
    """URDF mirroring the MJCF: -z chain, unit-mass links, COM at link end, y-axis hinges."""
    parts = [f'<?xml version="1.0" encoding="utf-8"?>\n<robot name="{name}">\n']

    parts.append(
        '  <link name="link_0">\n'
        '    <inertial><origin xyz="0 0 0"/><mass value="0"/>'
        '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>\n'
        '  </link>\n'
    )
    for i in range(1, n_links + 1):
        parts.append(
            f'  <link name="link_{i}">\n'
            f'    <inertial><origin xyz="0 0 -{COM_OFFSET}"/><mass value="{LINK_MASS}"/>'
            '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>\n'
            '  </link>\n'
        )
    parts.append(
        '  <link name="link_end">\n'
        '    <inertial><origin xyz="0 0 0"/><mass value="0"/>'
        '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>\n'
        '  </link>\n'
    )

    parts.append(
        '  <joint name="joint_1" type="revolute">\n'
        '    <parent link="link_0"/><child link="link_1"/>\n'
        '    <origin xyz="0 0 0"/>\n'
        '    <axis xyz="0 1 0"/>\n'
        '    <limit lower="-99999" upper="99999" effort="99999" velocity="99999"/>\n'
        '  </joint>\n'
    )
    for i in range(2, n_links + 1):
        parts.append(
            f'  <joint name="joint_{i}" type="revolute">\n'
            f'    <parent link="link_{i-1}"/><child link="link_{i}"/>\n'
            f'    <origin xyz="0 0 -{LINK_LENGTH}"/>\n'
            '    <axis xyz="0 1 0"/>\n'
            '    <limit lower="-99999" upper="99999" effort="99999" velocity="99999"/>\n'
            '  </joint>\n'
        )
    parts.append(
        '  <joint name="joint_end" type="fixed">\n'
        f'    <parent link="link_{n_links}"/><child link="link_end"/>\n'
        f'    <origin xyz="0 0 -{LINK_LENGTH}"/>\n'
        '  </joint>\n'
    )
    parts.append('</robot>\n')
    return "".join(parts)


def solve_one_phlame(N: int, x_init: np.ndarray, x_target: np.ndarray,
                     fp_urdf: str, dt: float, run_name: str):
    """Return (q, qd, qdd, t_solve) with shapes (N, T+1) for q/qd/qdd."""
    j_type = (2 * np.ones((N, 1))).astype(np.double, order='F')
    X0 = np.hstack([x_init[:N], x_init[N:]]).reshape(-1, 1).astype(np.double, order='F')
    Xf = np.hstack([x_target[:N], x_target[N:]]).reshape(-1, 1).astype(np.double, order='F')

    pset = ParameterSetBase(
        P_NODES, N, X0, Xf, run_name, S_MAX, K_PENALTY,
        ABS_TOL, REL_TOL, METHOD_NAME, MAX_STEPS, NS_POINTS,
        j_type, fp_urdf,
    )
    result = Experiment.run_single_static(
        pset, TIMEOUT_SEC, mode="general",
        use_jacobian=False, print_debug=False,
    )

    post = PostAghf(result)
    ps_values_fin = post.first_result.sol[-1, :].reshape(-1, 1)
    n_points = int(PHLAME_T_SIM / dt) + 1
    t_interp = np.linspace(-1, 1, n_points)
    q, qd, qdd = post.get_q_qd_qdd(ps_values_fin, t_interp)
    return q, qd, qdd, result.t_solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    N = cfg.NQ

    if abs(cfg.SIM_DURATION - PHLAME_T_SIM) > 1e-6:
        print(f"ERROR: cfg.SIM_DURATION={cfg.SIM_DURATION} but PHLAME requires {PHLAME_T_SIM}")
        sys.exit(1)

    urdf_str = generate_matched_urdf(N, name=cfg.PLANT_NAME)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(urdf_str)
    tmp.close()
    fp_urdf = tmp.name
    print(f"generated matched URDF: {fp_urdf}  (N={N})")

    x_inits, x_targets = sample_targets(cfg, cfg.N_TRAJECTORIES, cfg.TRAJECTORY_SAMPLE_SEED)

    n_steps_plus_1 = int(PHLAME_T_SIM / cfg.TIMESTEP) + 1
    x_refs = np.zeros((cfg.N_TRAJECTORIES, n_steps_plus_1, 2 * N))
    u_refs = np.zeros((cfg.N_TRAJECTORIES, n_steps_plus_1 - 1, N))
    converged_flags = np.zeros(cfg.N_TRAJECTORIES, dtype=bool)
    solve_times = np.zeros(cfg.N_TRAJECTORIES)

    print(f"sampling {cfg.N_TRAJECTORIES} trajectories with PHLAME "
          f"({N} links, T={PHLAME_T_SIM}s, dt={cfg.TIMESTEP})")

    for i in range(cfg.N_TRAJECTORIES):
        try:
            q, qd, qdd, t_solve = solve_one_phlame(
                N, x_inits[i], x_targets[i], fp_urdf,
                dt=cfg.TIMESTEP,
                run_name=f"{cfg.PLANT_NAME}_traj_{i:03d}",
            )
            x_refs[i, :, :N] = q.T
            x_refs[i, :, N:] = qd.T
            converged_flags[i] = True
            solve_times[i] = t_solve
            print(f"  traj {i+1:3d}/{cfg.N_TRAJECTORIES}  OK  ({t_solve:.2f}s)")
        except Exception as e:
            print(f"  traj {i+1:3d}/{cfg.N_TRAJECTORIES}  FAIL: {type(e).__name__}: {e}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = cfg.OUTPUT_DIR / "trajectories.npz"
    np.savez(
        output_path,
        x_refs=x_refs,
        u_refs=u_refs,
        x_inits=x_inits,
        x_targets=x_targets,
        converged=converged_flags,
        solve_times=solve_times,
    )
    n_ok = int(converged_flags.sum())
    avg_t = float(solve_times[converged_flags].mean()) if n_ok > 0 else 0.0
    total_t = float(solve_times.sum())
    print(f"\nsaved {output_path}")
    print(f"converged {n_ok}/{cfg.N_TRAJECTORIES}, "
          f"avg solve time {avg_t:.2f}s, total wall time {total_t:.1f}s")

    os.unlink(fp_urdf)


if __name__ == "__main__":
    main()
