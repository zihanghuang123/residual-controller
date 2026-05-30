"""Train pure MLP residual controller via BPTT through MJX under DR.

Random (trajectory, t0, plant) per iteration, BPTT through n_rollout MJX steps, tracking + control regularization loss. No theta information of any kind — the controller sees only the history and the reference window.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from double_pendulum import config as cfg
from lib import rollout, training
from lib.networks import MLPPureController

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
PARAMS_PATH = OUTPUT_DIR / "pure_params.pkl"
LOSS_PATH = OUTPUT_DIR / "pure_loss_history.npy"


def init_network(key):
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
    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(init_key)

    training.train_mlp_controller(
        cfg, cfg.PURE,
        params=params,
        build_controller_fn=make_build_controller_fn(network),
        traj_path=TRAJ_PATH, params_path=PARAMS_PATH, loss_path=LOSS_PATH,
        key=key,
    )


if __name__ == "__main__":
    main()
