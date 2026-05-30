"""Train the controller with frozen theta estimator (Stage 2 of two-model approach).

Loads the theta estimator params from Stage 1 and trains the controller end-to-end via BPTT through MJX under DR. At each rollout step the estimator predicts theta_hat from the current (x, u) history, and the controller sees theta_hat as an extra input alongside history + reference.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from lib import rollout, training
from lib.networks import MLPController, MLPThetaEstimator


def load_theta_params(theta_params_path):
    with open(theta_params_path, "rb") as f:
        return pickle.load(f)


def init_network(cfg, key):
    network = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)
    in_dim = training.mlp_residual_input_dim(cfg, cfg.CONTROLLER["n_history"], with_theta=True)
    params = network.init(key, jnp.zeros(in_dim))
    return network, params


def make_build_controller_fn(controller_network, theta_network, theta_params):
    """Two-model: estimator's theta_hat (from history) is fed to the controller. True theta is ignored."""
    def build(controller_params, _theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            theta_input = rollout.make_network_input(x_hist_full, u_hist)
            theta_hat = theta_network.apply(theta_params, theta_input)
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window, u_ref_window,
                theta_estimate=theta_hat,
            )
            return controller_network.apply(controller_params, controller_input)
        return controller_fn
    return build


def main():
    cfg = training.load_config()

    print("loading frozen theta estimator ...")
    theta_params = load_theta_params(cfg.OUTPUT_DIR / "theta_params.pkl")
    theta_network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)

    print("initializing controller network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    controller_network, controller_params = init_network(cfg, init_key)

    training.train_mlp_controller(
        cfg, cfg.CONTROLLER,
        params=controller_params,
        build_controller_fn=make_build_controller_fn(controller_network, theta_network, theta_params),
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        params_path=cfg.OUTPUT_DIR / "controller_params.pkl",
        loss_path=cfg.OUTPUT_DIR / "controller_loss_history.npy",
        key=key,
    )


if __name__ == "__main__":
    main()
