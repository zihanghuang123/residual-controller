"""Train pure MLP residual controller via BPTT through MJX under DR.

Random (trajectory, t0, plant) per iteration, BPTT through n_rollout MJX steps, tracking + control regularization loss. No theta information of any kind — the controller sees only the history and the reference window.

Every EVAL_EVERY iterations: runs eval_mlp_residual, appends a row to <output>/pure_eval_log.csv, saves best-endpoint params to <output>/pure_params_best.pkl.
"""

import functools
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from lib import evaluation, rollout, training
from lib.networks import MLPPureController

EVAL_EVERY = 500   # iterations between periodic evals; None disables


def init_network(cfg, key):
    network = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
    in_dim = training.mlp_residual_input_dim(cfg, cfg.PURE["n_history"], with_theta=False)
    params = network.init(key, jnp.zeros(in_dim))
    return network, params


def make_build_controller_fn(network):
    """Pure: controller sees only history + reference. theta is ignored."""
    def build(params, _theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
            return network.apply(params, net_in)
        return controller_fn
    return build


def main():
    cfg = training.load_config()

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(cfg, init_key)

    params_path = cfg.OUTPUT_DIR / "pure_params.pkl"
    if params_path.exists():
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        print(f"warm-starting from {params_path}")
    else:
        print("no existing pure_params.pkl; training from scratch")

    build_pure = make_build_controller_fn(network)
    pd_factory = lambda _theta: lambda *_: jnp.zeros(cfg.NU)

    def make_controllers(current_params):
        return {"pd": pd_factory, "pure": functools.partial(build_pure, current_params)}

    eval_callback = evaluation.make_eval_callback(
        cfg,
        make_controllers=make_controllers,
        target_name="pure",
        w=cfg.PURE["n_history"],
        csv_path=cfg.OUTPUT_DIR / "pure_eval_log.csv",
        best_params_path=cfg.OUTPUT_DIR / "pure_params_best.pkl",
    )

    training.train_mlp_controller(
        cfg, cfg.PURE,
        params=params,
        build_controller_fn=build_pure,
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        params_path=params_path,
        loss_path=cfg.OUTPUT_DIR / "pure_loss_history.npy",
        opt_state_path=cfg.OUTPUT_DIR / "pure_opt_state.pkl",
        key=key,
        eval_callback=eval_callback,
        eval_every=EVAL_EVERY,
    )


if __name__ == "__main__":
    main()
