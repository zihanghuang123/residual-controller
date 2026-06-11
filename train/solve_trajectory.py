"""Solve trajectory optimization on the nominal plant via Crocoddyl FDDP.

Generates N_TRAJECTORIES trajectories from sampled (x_init, x_target) pairs and stacks them into a single .npz at <cfg.OUTPUT_DIR>/trajectories.npz.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import crocoddyl
import numpy as np
import pinocchio as pin

from lib import training
from lib.sampling import sample_targets


def build_model(cfg) -> pin.Model:
    """Load the Pinocchio model from the plant's MJCF."""
    return pin.buildModelFromMJCF(str(cfg.MODEL_PATH))


def build_problem(cfg, model: pin.Model,
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

    return crocoddyl.ShootingProblem(x_init, [running_action] * cfg.N_STEPS, terminal_action)


def solve_one(cfg, model: pin.Model, x_init: np.ndarray, x_target: np.ndarray):
    """Solve one TO. Returns (xs, us, converged)."""
    problem = build_problem(cfg, model, x_init, x_target)
    solver = crocoddyl.SolverFDDP(problem)

    nu = problem.runningModels[0].nu
    xs_init = [(1.0 - k / cfg.N_STEPS) * x_init + (k / cfg.N_STEPS) * x_target
        for k in range(cfg.N_STEPS + 1)]
    us_init = [np.zeros(nu)] * cfg.N_STEPS
    converged = solver.solve(xs_init, us_init, maxiter=200)

    xs = np.array(solver.xs)
    us = np.array(solver.us)
    return xs, us, converged


def main() -> None:
    cfg = training.load_config()
    output_path = cfg.OUTPUT_DIR / "trajectories.npz"

    print(f"loading Pinocchio model from {cfg.MODEL_PATH}")
    model = build_model(cfg)
    print(f"  nq={model.nq}  nv={model.nv}")

    print(f"sampling {cfg.N_TRAJECTORIES} (x_init, x_target) pairs ...")
    x_inits, x_targets = sample_targets(cfg, cfg.N_TRAJECTORIES, cfg.TRAJECTORY_SAMPLE_SEED)

    x_refs = []
    u_refs = []
    converged_flags = []

    for i in range(cfg.N_TRAJECTORIES):
        xs, us, converged = solve_one(cfg, model, x_inits[i], x_targets[i])
        x_refs.append(xs)
        u_refs.append(us)
        converged_flags.append(converged)
        status = "OK  " if converged else "FAIL"
        print(f"  trajectory {i + 1:3d}/{cfg.N_TRAJECTORIES}  {status}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x_refs=np.stack(x_refs),                  # (N, T+1, 2*nq)
        u_refs=np.stack(u_refs),                  # (N, T,   nu)
        x_inits=x_inits,                          # (N, 2*nq)
        x_targets=x_targets,                      # (N, 2*nq)
        converged=np.array(converged_flags),      # (N,)
    )
    n_ok = sum(converged_flags)
    print(f"saved {output_path}")
    print(f"converged: {n_ok}/{cfg.N_TRAJECTORIES}")


if __name__ == "__main__":
    main()
