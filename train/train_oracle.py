"""Train the oracle controller: same network as the two-model controller, but fed the ground-truth theta.

Upper bound on how much access to plant parameters could possibly help. Estimator is bypassed entirely — the controller sees the true theta sampled at rollout time. If oracle ≈ pure, theta information doesn't help on this task and no estimator could close the gap. If oracle ≫ pure, that gap is the budget the two-model approach has to recover.

Every EVAL_EVERY iterations: runs eval_mlp_residual, appends a row to <output>/oracle_eval_log.csv, saves best-endpoint params to <output>/oracle_params_best.pkl.
"""

import functools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from lib import evaluation, rollout, training
from lib.networks import MLPController

EVAL_EVERY = 500   # iterations between periodic evals; None disables


def init_network(cfg, key):
    network = MLPController(hidden_sizes=cfg.ORACLE["hidden_sizes"], out_dim=cfg.NU)
    in_dim = training.mlp_residual_input_dim(cfg, cfg.ORACLE["n_history"], with_theta=True)
    params = network.init(key, jnp.zeros(in_dim))
    return network, params


def make_build_controller_fn(network):
    """Oracle: true theta (sampled at rollout time) is fed to the controller."""
    def build(params, theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window, u_ref_window,
                theta_estimate=theta,
            )
            return network.apply(params, controller_input)
        return controller_fn
    return build


def main():
    cfg = training.load_config()

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(cfg, init_key)

    build_oracle = make_build_controller_fn(network)
    pd_factory = lambda _theta: lambda *_: jnp.zeros(cfg.NU)

    def make_controllers(current_params):
        return {"pd": pd_factory, "oracle": functools.partial(build_oracle, current_params)}

    eval_callback = evaluation.make_eval_callback(
        cfg,
        make_controllers=make_controllers,
        target_name="oracle",
        w=cfg.ORACLE["n_history"],
        csv_path=cfg.OUTPUT_DIR / "oracle_eval_log.csv",
        best_params_path=cfg.OUTPUT_DIR / "oracle_params_best.pkl",
    )

    training.train_mlp_controller(
        cfg, cfg.ORACLE,
        params=params,
        build_controller_fn=build_oracle,
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        params_path=cfg.OUTPUT_DIR / "oracle_params.pkl",
        loss_path=cfg.OUTPUT_DIR / "oracle_loss_history.npy",
        opt_state_path=cfg.OUTPUT_DIR / "oracle_opt_state.pkl",
        key=key,
        eval_callback=eval_callback,
        eval_every=EVAL_EVERY,
    )


if __name__ == "__main__":
    main()
