"""Train pure RNN (GRU) residual controller via BPTT through MJX under DR.

The hidden state is the history, so there's no n_history knob and no padded
history buffers.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib import evaluation, losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import GRUPureController, gru_initial_state

EVAL_EVERY = 500   # closed-loop eval cadence; None disables
N_EVAL = 200


def init_network(cfg, key):
    network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    nx = 2 * cfg.NQ
    step_in_dim = nx + cfg.NU + nx + cfg.NU  # [x_curr, u_prev, x_ref, u_ref]
    dummy_h = gru_initial_state(cfg.PURE_RNN["hidden_sizes"])
    params = network.init(key, dummy_h, jnp.zeros(step_in_dim))
    return network, params


def make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs):
    H = cfg.PURE_RNN["n_rollout"]
    nq = cfg.NQ
    hidden_sizes = cfg.PURE_RNN["hidden_sizes"]
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def loss_fn(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (H, cfg.NU))
        x_ref_for_loss = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H + 1, 2 * nq))

        def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
            return network.apply(params, h, rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref))

        xs, _us, vs, x_final = rollout.rollout_rnn(
            mjx_model, x_ref_window[0], x_ref_window, u_ref_window,
            gru_initial_state(hidden_sizes), controller_fn, kp, kd, H)
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        return losses.tracking_loss(xs_full, x_ref_for_loss, nq) + cfg.PURE_RNN["alpha_reg"] * losses.reg_loss(vs)

    return loss_fn


def main():
    cfg = training.load_config()
    H = cfg.PURE_RNN["n_rollout"]

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
        optax.clip_by_global_norm(cfg.PURE_RNN["grad_clip_norm"]),
        optax.adam(cfg.PURE_RNN["lr"]),
    )
    opt_state = optimizer.init(params)

    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    opt_state_path = cfg.OUTPUT_DIR / "pure_rnn_opt_state.pkl"
    if params_path.exists():
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        print(f"loaded best params from {params_path}")
    else:
        print("no best params; params from scratch")
    if opt_state_path.exists():
        with open(opt_state_path, "rb") as f:
            opt_state = pickle.load(f)
        print(f"loaded best opt_state from {opt_state_path}")
    else:
        print("no best opt_state; optimizer from scratch")

    loss_fn = make_loss_fn(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs)
    train_step = training.make_train_step(loss_fn, optimizer)

    rnn_cb = evaluation.make_rnn_eval_callback(
        cfg, network, gru_initial_state(cfg.PURE_RNN["hidden_sizes"]),
        x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
        csv_path=cfg.OUTPUT_DIR / "pure_rnn_eval_log.csv",
        best_params_path=params_path, best_opt_state_path=opt_state_path, n_eval=N_EVAL)
    eval_callback = lambda p, o, it: rnn_cb(p, it, o)   # pass opt_state so the best opt_state is saved too

    print(f"training: {cfg.PURE_RNN['n_iterations']} iterations, batch={cfg.PURE_RNN['batch_size']}, "
          f"H={H}, hidden={cfg.PURE_RNN['hidden_sizes']}")
    params, opt_state, loss_history = training.training_loop(
        key, params, opt_state, train_step,
        batch_size=cfg.PURE_RNN["batch_size"], n_iterations=cfg.PURE_RNN["n_iterations"],
        n_traj=n_traj, t0_max=T - H + 1,
        eval_callback=eval_callback, eval_every=EVAL_EVERY)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = cfg.OUTPUT_DIR / "pure_rnn_loss_history.npy"
    if not params_path.exists():   # fallback only: no eval ever ran, so save the final iterate
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
        with open(opt_state_path, "wb") as f:
            pickle.dump(opt_state, f)
    np.save(loss_path, loss_history)
    print(f"saved {loss_path}; best closed-loop params at {params_path}")


if __name__ == "__main__":
    main()
