"""Supervised GRU residual training: full-sequence BPTT on inverse-dynamics labels at perturbed states."""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib import evaluation, networks, rollout, training

RNN_WARMUP = 50    # skip the cold-start transient (zero-init h) when scoring the loss
LOG_EVERY = 100
EVAL_EVERY = 2000  # closed-loop eval cadence; None disables
N_EVAL = 200


def main():
    cfg = training.load_config()
    nq, nu = cfg.NQ, cfg.NU

    print("loading dataset ...")
    data = np.load(cfg.OUTPUT_DIR / "supervised_dataset.npz")
    x_refs = jnp.asarray(data["x_refs"])               # (N, T+1, 2*nq)
    u_refs = jnp.asarray(data["u_refs"])               # (N, T, nu)
    x_pert = jnp.asarray(data["x_perturbed"])          # (N, T, 2*nq), drifted states
    labels = jnp.asarray(data["u_residual_labels"])    # (N, M, T, nu)

    N = x_refs.shape[0]
    T = u_refs.shape[1]
    M = labels.shape[1]
    print(f"  N={N} trajectories, M={M} thetas, T={T} steps")

    hp = cfg.PURE_RNN
    hidden_sizes = hp["hidden_sizes"]
    batch_size = hp["batch_size"]
    lr = hp["lr"]
    n_iter = hp.get("n_iterations_supervised", hp["n_iterations"])
    grad_clip = hp["grad_clip_norm"]
    print(f"  hidden={hidden_sizes}, batch_size={batch_size}, lr={lr}, n_iter={n_iter}")
    print(f"  full-sequence BPTT over T={T}; lower batch_size if OOM")

    network = networks.GRUPureController(hidden_sizes=hidden_sizes, out_dim=nu)
    h0 = networks.gru_initial_state(hidden_sizes)
    dummy_x = rollout.make_rnn_step_input(jnp.zeros(2 * nq), jnp.zeros(nu), jnp.zeros(2 * nq), jnp.zeros(nu))

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, h0, dummy_x)

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optimizer.init(params)

    def build_sequence(x_pert, x_refs, u_refs, labels, traj_idx, theta_idx):
        x_cur_seq = x_pert[traj_idx]                  # (T, 2*nq) drifted state the net sees
        x_ref_seq = x_refs[traj_idx, :T]             # (T, 2*nq) reference
        u_ref_seq = u_refs[traj_idx]
        label_seq = labels[traj_idx, theta_idx]
        applied = u_ref_seq + label_seq              # control consistent with this theta
        u_prev_seq = jnp.concatenate([jnp.zeros((1, nu)), applied[:-1]], axis=0)
        step_inputs = jax.vmap(rollout.make_rnn_step_input)(x_cur_seq, u_prev_seq, x_ref_seq, u_ref_seq)
        return step_inputs, label_seq

    def run_sequence(params, step_inputs):
        def step(h, x):
            return network.apply(params, h, x)
        _, preds = jax.lax.scan(step, h0, step_inputs)
        return preds

    def per_example_loss(params, x_pert, x_refs, u_refs, labels, traj_idx, theta_idx):
        step_inputs, label_seq = build_sequence(x_pert, x_refs, u_refs, labels, traj_idx, theta_idx)
        preds = run_sequence(params, step_inputs)
        return jnp.mean((preds[RNN_WARMUP:] - label_seq[RNN_WARMUP:]) ** 2)

    # Arrays passed as args (None axis), not closed over, so XLA doesn't bake the dataset in.
    batched_loss = jax.vmap(per_example_loss, in_axes=(None, None, None, None, None, 0, 0))

    def loss_fn(params, x_pert, x_refs, u_refs, labels, traj_idxs, theta_idxs):
        return jnp.mean(batched_loss(params, x_pert, x_refs, u_refs, labels, traj_idxs, theta_idxs))

    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, opt_state, x_pert, x_refs, u_refs, labels, traj_idxs, theta_idxs):
        loss, grads = grad_fn(params, x_pert, x_refs, u_refs, labels, traj_idxs, theta_idxs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("building closed-loop eval ...")
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)
    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    eval_callback = evaluation.make_rnn_eval_callback(
        cfg, network, h0, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
        csv_path=cfg.OUTPUT_DIR / "pure_rnn_eval_log.csv",
        best_params_path=params_path,
        n_eval=N_EVAL)

    print(f"training {n_iter} iterations ...")
    loss_history = np.zeros(n_iter)
    for i in range(n_iter):
        key, k_traj, k_theta = jax.random.split(key, 3)
        traj_idxs = jax.random.randint(k_traj, (batch_size,), 0, N)
        theta_idxs = jax.random.randint(k_theta, (batch_size,), 0, M)

        params, opt_state, loss = train_step(
            params, opt_state, x_pert, x_refs, u_refs, labels, traj_idxs, theta_idxs)
        loss_history[i] = float(loss)

        if (i + 1) % LOG_EVERY == 0:
            print(f"  iter {i + 1:5d}/{n_iter}  loss={loss:.6f}")
        if EVAL_EVERY and (i + 1) % EVAL_EVERY == 0:
            eval_callback(params, i + 1)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = cfg.OUTPUT_DIR / "pure_rnn_loss_history.npy"
    if not params_path.exists():   # no eval checkpoint saved; fall back to the final iterate
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
    np.save(loss_path, loss_history)
    print(f"saved {loss_path}; best closed-loop params at {params_path}")


if __name__ == "__main__":
    main()
