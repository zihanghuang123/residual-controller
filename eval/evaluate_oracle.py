"""Full hold-out eval, PD vs oracle controller.

Same N_EVAL_PLANTS / EVAL_SEED protocol as evaluate_pure.py. The oracle is the MLPController (same arch as the two-model controller) fed the true theta sampled at rollout time, bypassing the estimator.

Compare metrics_oracle.npz against metrics_pure.npz: if oracle reduces endpoint vs pure, theta info is worth fighting for. If oracle ≈ pure, pure is already at the achievable ceiling.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from lib import evaluation, rollout, training
from lib.networks import MLPController


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_pd_factory(cfg):
    def make_controller_fn(_theta):
        return lambda *_: jnp.zeros(cfg.NU)
    return make_controller_fn


def make_oracle_factory(network, params):
    def make_controller_fn(theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window, u_ref_window,
                theta_estimate=theta,
            )
            return network.apply(params, controller_input)
        return controller_fn
    return make_controller_fn


def main():
    cfg = training.load_config()

    print("loading oracle params + building network ...")
    oracle_params = load_pkl(cfg.OUTPUT_DIR / "oracle_params.pkl")
    oracle_network = MLPController(hidden_sizes=cfg.ORACLE["hidden_sizes"], out_dim=cfg.NU)

    controllers = {
        "pd": make_pd_factory(cfg),
        "oracle": make_oracle_factory(oracle_network, oracle_params),
    }

    results = evaluation.eval_mlp_residual(
        cfg, controllers,
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        metrics_path=cfg.OUTPUT_DIR / "metrics_oracle.npz",
        w=cfg.ORACLE["n_history"],
    )

    ep_pd = results["pd"][0].mean()
    ep_oracle = results["oracle"][0].mean()
    print(f"\noracle reduces mean endpoint by {100 * (1 - ep_oracle / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_oracle:.3f})")


if __name__ == "__main__":
    main()
