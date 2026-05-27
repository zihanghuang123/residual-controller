"""Solve trajectory optimization on the nominal plant via Crocoddyl FDDP.

Generates N_TRAJECTORIES trajectories from sampled (x_init, x_target) pairs and stacks them into a single .npz at outputs/double_pendulum/trajectories.npz.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import crocoddyl
import numpy as np
import pinocchio as pin

from double_pendulum import config as cfg

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
OUTPUT_PATH = OUTPUT_DIR / "trajectories.npz"


def build_model() -> pin.Model:
    """Load the Pinocchio model from the plant's MJCF."""
    return pin.buildModelFromMJCF(str(cfg.MODEL_PATH))


def build_problem(model: pin.Model,
                  x_init: np.ndarray,
                  x_target: np.ndarray) -> crocoddyl.ShootingProblem:
    """Assemble the Crocoddyl shooting problem for one (x_init, x_target) pair."""
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    nu = actuation.nu

    running_cost = crocoddyl.CostModelSum(state, nu)
    running_cost.addCost(
        "x",
        crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state, x_target, nu)),
        cfg.TO_COST_X_RUNNING,
    )
    running_cost.addCost(
        "u",
        crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)),
        cfg.TO_COST_U_RUNNING,
    )

    terminal_cost = crocoddyl.CostModelSum(state, nu)
    terminal_cost.addCost(
        "x",
        crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state, x_target, nu)),
        cfg.TO_COST_X_TERMINAL,
    )

    running_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, running_cost)
    terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_cost)
    running_action = crocoddyl.IntegratedActionModelEuler(running_diff, cfg.TIMESTEP)
    terminal_action = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

    # Position in ShootingProblem assigns the role: 3rd arg is the terminal action.
    return crocoddyl.ShootingProblem(x_init, [running_action] * cfg.N_STEPS, terminal_action)


def solve_one(model: pin.Model, x_init: np.ndarray, x_target: np.ndarray):
    """Solve one TO. Returns (xs, us, converged)."""
    problem = build_problem(model, x_init, x_target)
    solver = crocoddyl.SolverFDDP(problem)

    nu = problem.runningModels[0].nu
    xs_init = [(1.0 - k / cfg.N_STEPS) * x_init + (k / cfg.N_STEPS) * x_target
        for k in range(cfg.N_STEPS + 1)]
    us_init = [np.zeros(nu)] * cfg.N_STEPS
    converged = solver.solve(xs_init, us_init, maxiter=200)

    xs = np.array(solver.xs)
    us = np.array(solver.us)
    return xs, us, converged


def sample_targets(n_trajectories: int, seed: int):
    """Sample N (x_init, x_target) pairs uniformly from config ranges. qvel = 0 for both."""
    rng = np.random.default_rng(seed)

    qpos_init_lo, qpos_init_hi = cfg.INITIAL_QPOS_RANGE
    qpos_target_lo, qpos_target_hi = cfg.TARGET_QPOS_RANGE

    qpos_inits = rng.uniform(qpos_init_lo, qpos_init_hi, size=(n_trajectories, cfg.NQ))
    qpos_targets = rng.uniform(qpos_target_lo, qpos_target_hi, size=(n_trajectories, cfg.NQ))

    x_inits = np.hstack([qpos_inits, np.zeros_like(qpos_inits)])
    x_targets = np.hstack([qpos_targets, np.zeros_like(qpos_targets)])
    return x_inits, x_targets


def main() -> None:
    print(f"loading Pinocchio model from {cfg.MODEL_PATH}")
    model = build_model()
    print(f"  nq={model.nq}  nv={model.nv}")

    print(f"sampling {cfg.N_TRAJECTORIES} (x_init, x_target) pairs ...")
    x_inits, x_targets = sample_targets(cfg.N_TRAJECTORIES, cfg.TRAJECTORY_SAMPLE_SEED)

    x_refs = []
    u_refs = []
    converged_flags = []

    for i in range(cfg.N_TRAJECTORIES):
        xs, us, converged = solve_one(model, x_inits[i], x_targets[i])
        x_refs.append(xs)
        u_refs.append(us)
        converged_flags.append(converged)
        status = "OK  " if converged else "FAIL"
        print(f"  trajectory {i + 1:3d}/{cfg.N_TRAJECTORIES}  {status}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        x_refs=np.stack(x_refs),                  # (N, T+1, 2*nq)
        u_refs=np.stack(u_refs),                  # (N, T,   nu)
        x_inits=x_inits,                          # (N, 2*nq)
        x_targets=x_targets,                      # (N, 2*nq)
        converged=np.array(converged_flags),      # (N,)
    )
    n_ok = sum(converged_flags)
    print(f"saved {OUTPUT_PATH}")
    print(f"converged: {n_ok}/{cfg.N_TRAJECTORIES}")


if __name__ == "__main__":
    main()
