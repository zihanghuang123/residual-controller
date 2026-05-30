"""Train pure MLP residual controller via BPTT through MJX under DR.

Random (trajectory, t0, plant) per iteration, BPTT through n_rollout MJX steps, tracking + control regularization loss. No theta information of any kind — the controller sees only the history and the reference window.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import optax

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
    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(TRAJ_PATH)
    n_traj, T_plus_1, _ = x_refs.shape
    T = T_plus_1 - 1
    H = cfg.PURE["n_rollout"]
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(init_key)

    print("building optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.PURE["grad_clip_norm"]),
        optax.adam(cfg.PURE["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = training.make_mlp_residual_loss(
        cfg, cfg.PURE,
        mjx_model_nominal, nominal_body_mass,
        x_refs, u_refs,
        build_controller_fn=make_build_controller_fn(network),
    )
    train_step = training.make_train_step(loss_fn, optimizer)

    batch_size = cfg.PURE["batch_size"]
    n_iterations = cfg.PURE["n_iterations"]
    print(f"training: {n_iterations} iterations, batch={batch_size}, H={H}, w={cfg.PURE['n_history']}")
    params, loss_history = training.training_loop(
        key, params, opt_state, train_step,
        batch_size=batch_size, n_iterations=n_iterations,
        n_traj=n_traj, t0_max=T - H + 1,
    )

    training.save_results(params, loss_history, PARAMS_PATH, LOSS_PATH)


if __name__ == "__main__":
    main()
