"""Supervised MLP training on RNEA-computed residual labels. No BPTT, no MJX rollouts."""

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from lib import rollout, training
from lib.networks import MLPPureController


def load_config(config_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("plant_cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    nq, nu = cfg.NQ, cfg.NU

    print("loading dataset ...")
    data = np.load(cfg.OUTPUT_DIR / "supervised_dataset.npz")
    x_refs = jnp.asarray(data["x_refs"])               # (N, T+1, 2*nq)
    u_refs = jnp.asarray(data["u_refs"])               # (N, T, nu)
    labels = jnp.asarray(data["u_residual_labels"])    # (N, M, T, nu)

    N, T_plus_1, _ = x_refs.shape
    T = u_refs.shape[1]
    M = labels.shape[1]
    print(f"  N={N} trajectories, M={M} thetas, T={T} steps")

    hp = cfg.PURE
    w = hp["n_history"]
    batch_size = hp["batch_size"]
    lr = hp["lr"]
    n_iter = hp.get("n_iterations_supervised", hp["n_iterations"])
    grad_clip = hp["grad_clip_norm"]
    print(f"  w={w}, batch_size={batch_size}, lr={lr}, n_iter={n_iter}")

    network = MLPPureController(hidden_sizes=hp["hidden_sizes"], out_dim=nu)
    in_dim = training.mlp_residual_input_dim(cfg, w, with_theta=False)

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, jnp.zeros(in_dim))

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr),
    )
    opt_state = optimizer.init(params)

    t_min, t_max = w, T - w - 1

    def build_example(traj_idx, theta_idx, t):
        x_hist_full = jax.lax.dynamic_slice(x_refs[traj_idx], (t - w, 0), (w + 1, 2 * nq))
        u_nom_hist = jax.lax.dynamic_slice(u_refs[traj_idx], (t - w, 0), (w, nu))
        u_resid_hist = jax.lax.dynamic_slice(labels[traj_idx, theta_idx], (t - w, 0), (w, nu))
        u_hist = u_nom_hist + u_resid_hist
        x_ref_window = jax.lax.dynamic_slice(x_refs[traj_idx], (t, 0), (w + 1, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[traj_idx], (t, 0), (w + 1, nu))
        net_in = rollout.make_network_input(
            x_hist_full, u_hist, x_ref_window, u_ref_window,
        )
        return net_in, labels[traj_idx, theta_idx, t]

    def per_example_loss(params, traj_idx, theta_idx, t):
        net_in, label = build_example(traj_idx, theta_idx, t)
        pred = network.apply(params, net_in)
        return jnp.mean((pred - label) ** 2)

    batched_loss = jax.vmap(per_example_loss, in_axes=(None, 0, 0, 0))

    def loss_fn(params, traj_idxs, theta_idxs, ts):
        return jnp.mean(batched_loss(params, traj_idxs, theta_idxs, ts))

    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, opt_state, traj_idxs, theta_idxs, ts):
        loss, grads = grad_fn(params, traj_idxs, theta_idxs, ts)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print(f"training {n_iter} iterations ...")
    loss_history = np.zeros(n_iter)
    for i in range(n_iter):
        key, k_traj, k_theta, k_t = jax.random.split(key, 4)
        traj_idxs = jax.random.randint(k_traj, (batch_size,), 0, N)
        theta_idxs = jax.random.randint(k_theta, (batch_size,), 0, M)
        ts = jax.random.randint(k_t, (batch_size,), t_min, t_max + 1)

        params, opt_state, loss = train_step(params, opt_state, traj_idxs, theta_idxs, ts)
        loss_history[i] = float(loss)

        if (i + 1) % args.log_every == 0:
            print(f"  iter {i + 1:5d}/{n_iter}  loss={loss:.6f}")

    params_path = cfg.OUTPUT_DIR / "pure_params.pkl"
    loss_path = cfg.OUTPUT_DIR / "pure_loss_history.npy"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    np.save(loss_path, loss_history)
    print(f"saved {params_path}")
    print(f"saved {loss_path}")


if __name__ == "__main__":
    main()
