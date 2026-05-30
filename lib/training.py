"""Shared scaffolding for MLP residual controller training (BPTT through MJX under DR).

Used by train_pure / train_controller / train_oracle. Each script keeps only what's unique to it: paths, the network class, and a small `build_controller_fn(params, theta) -> controller_fn` factory describing how a single step's residual is computed.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx

from lib import losses, rollout
from lib.domain_randomization import apply_theta, sample_theta


def load_trajectories(traj_path: Path):
    """Load TO trajectories, keep only the converged ones."""
    data = np.load(traj_path)
    mask = data["converged"].astype(bool)
    x_refs = jnp.asarray(data["x_refs"][mask])    # (N, T+1, 2*nq)
    u_refs = jnp.asarray(data["u_refs"][mask])    # (N, T,   nu)
    return x_refs, u_refs


def build_mjx_model(model_path: Path):
    """Load the MuJoCo model and push it to device as an mjx.Model."""
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mjx_model = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model.body_mass)
    return mjx_model, nominal_body_mass


def pad_history(x_ref_t0, u_ref_t0, w):
    """Initial history buffers for MLP-style rollouts: pad with the window-start reference."""
    x_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_hist0 = jnp.tile(u_ref_t0, (w, 1))
    x_ref_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_ref_hist0 = jnp.tile(u_ref_t0, (w, 1))
    return x_hist0, u_hist0, x_ref_hist0, u_ref_hist0


def mlp_residual_input_dim(cfg, w, with_theta=False):
    """Flat input dim for an MLP residual controller: history + reference window (+ theta)."""
    nx = 2 * cfg.NQ
    base = (w + 1) * nx + w * cfg.NU + (w + 1) * nx + (w + 1) * cfg.NU
    return base + (cfg.THETA_DIM if with_theta else 0)


def make_mlp_residual_loss(
    cfg, cfg_dict,
    mjx_model_nominal, nominal_body_mass,
    x_refs, u_refs,
    build_controller_fn,
):
    """Single-rollout tracking + control-regularization loss for MLP residual controllers.

    build_controller_fn(params, theta) -> controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window) -> v
    """
    w = cfg_dict["n_history"]
    H = cfg_dict["n_rollout"]
    alpha_reg = cfg_dict["alpha_reg"]
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def loss_fn(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (H, cfg.NU))
        x_ref_for_loss = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H + 1, 2 * nq))

        x_init = x_ref_window[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = pad_history(
            x_ref_window[0], u_ref_window[0], w
        )

        controller_fn = build_controller_fn(params, theta)

        xs, _us, vs, x_final = rollout.rollout(
            mjx_model, x_init, x_ref_window, u_ref_window,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0,
            controller_fn, kp, kd, H,
        )
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        return losses.tracking_loss(xs_full, x_ref_for_loss, nq) + alpha_reg * losses.reg_loss(vs)

    return loss_fn


def make_train_step(loss_fn, optimizer):
    """Vmap loss over a batch, take grads, apply optimizer step. Returns a jit'd fn."""
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))

    def mean_loss(params, theta_keys, idxs, t0s):
        return jnp.mean(batched_loss(params, theta_keys, idxs, t0s))

    grad_fn = jax.value_and_grad(mean_loss)

    @jax.jit
    def train_step(params, opt_state, theta_keys, idxs, t0s):
        loss, grads = grad_fn(params, theta_keys, idxs, t0s)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def training_loop(
    key, params, opt_state, train_step,
    batch_size, n_iterations, n_traj, t0_max,
    log_every=50,
):
    """Random-(idx, t0, theta) sampling loop. Returns (params, loss_history)."""
    loss_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        key, idx_key, t0_key, *theta_keys = jax.random.split(key, batch_size + 3)
        idxs = jax.random.randint(idx_key, (batch_size,), 0, n_traj)
        t0s = jax.random.randint(t0_key, (batch_size,), 0, t0_max)
        theta_keys = jnp.stack(theta_keys)

        params, opt_state, loss = train_step(params, opt_state, theta_keys, idxs, t0s)
        loss_history[i] = float(loss)

        if i % log_every == 0 or i == n_iterations - 1:
            print(f"  iter {i:5d}  loss = {float(loss):.6f}")
    return params, loss_history


def save_results(params, loss_history, params_path: Path, loss_path: Path):
    """Pickle params + numpy-save loss history. Parent directories created as needed."""
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    np.save(loss_path, loss_history)
    print(f"saved {params_path}")
    print(f"saved {loss_path}")


def train_mlp_controller(
    cfg, cfg_dict,
    params, build_controller_fn,
    traj_path: Path, params_path: Path, loss_path: Path,
    key,
):
    """End-to-end MLP residual controller training.

    Caller pre-builds the network, initializes params, and constructs build_controller_fn (binding the network and any auxiliaries like a frozen theta estimator). Everything else — data, MJX model, optimizer, loss, train_step, loop, save — happens here.
    """
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories(traj_path)
    n_traj, T_plus_1, _ = x_refs.shape
    T = T_plus_1 - 1
    H = cfg_dict["n_rollout"]
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = build_mjx_model(cfg.MODEL_PATH)

    print("building optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg_dict["grad_clip_norm"]),
        optax.adam(cfg_dict["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = make_mlp_residual_loss(
        cfg, cfg_dict,
        mjx_model_nominal, nominal_body_mass,
        x_refs, u_refs,
        build_controller_fn=build_controller_fn,
    )
    train_step = make_train_step(loss_fn, optimizer)

    batch_size = cfg_dict["batch_size"]
    n_iterations = cfg_dict["n_iterations"]
    print(f"training: {n_iterations} iterations, batch={batch_size}, H={H}, w={cfg_dict['n_history']}")
    params, loss_history = training_loop(
        key, params, opt_state, train_step,
        batch_size=batch_size, n_iterations=n_iterations,
        n_traj=n_traj, t0_max=T - H + 1,
    )

    save_results(params, loss_history, params_path, loss_path)
    return params, loss_history
