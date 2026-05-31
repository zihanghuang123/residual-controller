"""Train the theta estimator, GRU variant (Stage 1 of two-model approach).

A PD-only rollout under DR produces an (x, u) sequence; a GRU streams over it and
its final output predicts theta. Per-step estimator input: (x_t, u_{t-1}).

Writes to theta_rnn_params.pkl so it never clobbers the MLP estimator's artifacts.
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
from lib.networks import GRUThetaEstimator, gru_initial_state


def init_network(cfg, key):
    network = GRUThetaEstimator(hidden_sizes=cfg.THETA_RNN["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    step_in_dim = 2 * cfg.NQ + cfg.NU  # [x_t, u_prev]
    dummy_h = gru_initial_state(cfg.THETA_RNN["hidden_sizes"])
    params = network.init(key, dummy_h, jnp.zeros(step_in_dim))
    return network, params


def make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs):
    H = cfg.THETA_RNN["n_rollout"]
    nq = cfg.NQ
    nu = cfg.NU
    hidden_sizes = cfg.THETA_RNN["hidden_sizes"]
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def pd_controller(h, x_curr, u_prev, x_ref, u_ref):
        return h, jnp.zeros(nu)

    def loss_fn(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (H, nu))

        xs, us, _vs, _xf = rollout.rollout_rnn(
            mjx_model, x_ref_window[0], x_ref_window, u_ref_window,
            jnp.zeros(1), pd_controller, kp, kd, H)
        u_prev_seq = jnp.concatenate([jnp.zeros((1, nu)), us[:-1]], axis=0)

        def est_step(h, step):
            x_t, u_prev = step
            return network.apply(params, h, rollout.make_rnn_step_input(x_t, u_prev))

        _, theta_preds = jax.lax.scan(est_step, gru_initial_state(hidden_sizes), (xs, u_prev_seq))
        return losses.theta_loss(theta_preds[-1], theta)

    return loss_fn


def main():
    cfg = training.load_config()
    H = cfg.THETA_RNN["n_rollout"]

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
        optax.clip_by_global_norm(cfg.THETA_RNN["grad_clip_norm"]),
        optax.adam(cfg.THETA_RNN["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs)
    train_step = training.make_train_step(loss_fn, optimizer)

    print(f"training: {cfg.THETA_RNN['n_iterations']} iterations, batch={cfg.THETA_RNN['batch_size']}, H={H}")
    params, loss_history = training.training_loop(
        key, params, opt_state, train_step,
        batch_size=cfg.THETA_RNN["batch_size"], n_iterations=cfg.THETA_RNN["n_iterations"],
        n_traj=n_traj, t0_max=T - H + 1)

    training.save_results(params, loss_history,
                          cfg.OUTPUT_DIR / "theta_rnn_params.pkl",
                          cfg.OUTPUT_DIR / "theta_rnn_loss_history.npy")


if __name__ == "__main__":
    main()
