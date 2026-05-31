"""Train the theta estimator (Stage 1 of two-model approach).

For each iteration: sample theta, perturb the plant, run a PD-only rollout for w+1 steps to generate a realistic (x, u) history under domain randomization, then train an MLP to predict theta from that history.

Estimator input: x history + u history only
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import optax

from lib import losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPThetaEstimator


def init_network(cfg, key):
    network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    w = cfg.THETA["n_history"]
    in_dim = (w + 1) * (2 * cfg.NQ) + w * cfg.NU
    params = network.init(key, jnp.zeros(in_dim))
    return network, params


def make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs):
    w = cfg.THETA["n_history"]
    n_rollout = w + 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def zero_controller(*_):
        return jnp.zeros(cfg.NU)

    def loss_fn(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (n_rollout, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (n_rollout, cfg.NU))
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = training.pad_history(
            x_ref_window[0], u_ref_window[0], w)

        xs, us, _vs, _x_final = rollout.rollout(
            mjx_model, x_ref_window[0], x_ref_window, u_ref_window,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, zero_controller, kp, kd, n_rollout)

        net_in = rollout.make_network_input(xs, us[:w])
        return losses.theta_loss(network.apply(params, net_in), theta)

    return loss_fn


def main():
    cfg = training.load_config()
    w = cfg.THETA["n_history"]
    n_rollout = w + 1

    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    n_traj, T_plus_1, _ = x_refs.shape
    T = T_plus_1 - 1
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(cfg, init_key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.THETA["grad_clip_norm"]),
        optax.adam(cfg.THETA["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs)
    train_step = training.make_train_step(loss_fn, optimizer)

    print(f"training: {cfg.THETA['n_iterations']} iterations, batch={cfg.THETA['batch_size']}, w={w}")
    params, loss_history = training.training_loop(
        key, params, opt_state, train_step,
        batch_size=cfg.THETA["batch_size"], n_iterations=cfg.THETA["n_iterations"],
        n_traj=n_traj, t0_max=T - n_rollout + 1)

    training.save_results(params, loss_history,
                          cfg.OUTPUT_DIR / "theta_params.pkl",
                          cfg.OUTPUT_DIR / "theta_loss_history.npy")


if __name__ == "__main__":
    main()
