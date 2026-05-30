"""Hold-out evaluation: PD vs pure vs two-model controllers over N_EVAL_PLANTS plants drawn from the DR distribution under EVAL_SEED.

Endpoint error = ||x_final - x_target|| (RMS, with angle wrap).
Tracking error = mean over time of ||x_t - x_ref_t||^2 (MSE, with angle wrap).
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from lib import evaluation, rollout, training
from lib.networks import MLPController, MLPPureController, MLPThetaEstimator


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_pd_factory(cfg):
    def make_controller_fn(_theta):
        return lambda *_: jnp.zeros(cfg.NU)
    return make_controller_fn


def make_pure_factory(network, params):
    def make_controller_fn(_theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
            return network.apply(params, net_in)
        return controller_fn
    return make_controller_fn


def make_two_model_factory(theta_network, theta_params, controller_network, controller_params):
    def make_controller_fn(_theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            theta_input = rollout.make_network_input(x_hist_full, u_hist)
            theta_hat = theta_network.apply(theta_params, theta_input)
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window, u_ref_window,
                theta_estimate=theta_hat,
            )
            return controller_network.apply(controller_params, controller_input)
        return controller_fn
    return make_controller_fn


def main():
    cfg = training.load_config()

    print("loading params + building networks ...")
    pure_params = load_pkl(cfg.OUTPUT_DIR / "pure_params.pkl")
    theta_params = load_pkl(cfg.OUTPUT_DIR / "theta_params.pkl")
    controller_params = load_pkl(cfg.OUTPUT_DIR / "controller_params.pkl")

    pure_network = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
    theta_network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    controller_network = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)

    controllers = {
        "pd": make_pd_factory(cfg),
        "pure": make_pure_factory(pure_network, pure_params),
        "two_model": make_two_model_factory(theta_network, theta_params, controller_network, controller_params),
    }

    evaluation.eval_mlp_residual(
        cfg, controllers,
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        metrics_path=cfg.OUTPUT_DIR / "metrics.npz",
        w=cfg.PURE["n_history"],
        name_width=12,
    )


if __name__ == "__main__":
    main()
