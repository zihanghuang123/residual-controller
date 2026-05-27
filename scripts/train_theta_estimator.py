"""Train the theta estimator (Stage 1 of two-model approach).

For each iteration: sample theta, perturb the plant, run a PD-only rollout for w+1 steps to generate a realistic (x, u) history under domain randomization, then train an MLP to predict theta from that history.

Estimator input: x history + u history only
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from mujoco import mjx

from double_pendulum import config as cfg
from lib import losses, rollout
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPThetaEstimator

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
PARAMS_PATH = OUTPUT_DIR / "theta_params.pkl"
LOSS_PATH = OUTPUT_DIR / "theta_loss_history.npy"


def load_trajectories():
    """Load TO trajectories, keep only the converged ones."""
    data = np.load(TRAJ_PATH)
    mask = data["converged"].astype(bool)
    x_refs = jnp.asarray(data["x_refs"][mask])    # (N, T+1, 2*nq)
    u_refs = jnp.asarray(data["u_refs"][mask])    # (N, T,   nu)
    return x_refs, u_refs


def build_mjx_model():
    """Load the MuJoCo model and push it to device as an mjx.Model."""
    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model.body_mass)
    return mjx_model, nominal_body_mass


def init_network(key):
    """Build the theta estimator and initialize params from a dummy input."""
    network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    w = cfg.THETA["n_history"]
    nx = 2 * cfg.NQ

    in_dim = (w + 1) * nx + w * cfg.NU
    dummy = jnp.zeros(in_dim)
    params = network.init(key, dummy)
    return network, params


def make_history_buffers(x_ref_t0, u_ref_t0, w):
    """Pad all four history buffers with the reference at the window start."""
    x_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_hist0 = jnp.tile(u_ref_t0, (w, 1))
    x_ref_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_ref_hist0 = jnp.tile(u_ref_t0, (w, 1))
    return x_hist0, u_hist0, x_ref_hist0, u_ref_hist0


def make_loss_fn(mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs):
    """Build the single-rollout loss fn closed over fixed data."""
    w = cfg.THETA["n_history"]
    n_rollout = w + 1 
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def loss_fn(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (n_rollout, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (n_rollout, cfg.NU))

        x_init = x_ref_window[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_window[0], u_ref_window[0], w
        )

        def zero_controller(*_):
            return jnp.zeros(cfg.NU)

        xs, us, _vs = rollout.rollout(
            mjx_model, x_init, x_ref_window, u_ref_window,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0,
            zero_controller, kp, kd, n_rollout,
        )

        net_in = rollout.make_network_input(xs, us[:w])
        theta_pred = network.apply(params, net_in)
        return losses.theta_loss(theta_pred, theta)

    return loss_fn


def make_train_step(loss_fn, optimizer):
    """Vmap the loss over a batch, take grads, apply the Adam step. Returns a jit'd fn."""
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


def main():
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories()
    n_traj, T_plus_1, _ = x_refs.shape
    T = T_plus_1 - 1
    w = cfg.THETA["n_history"]
    n_rollout = w + 1
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(init_key)

    print("building optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.THETA["grad_clip_norm"]),
        optax.adam(cfg.THETA["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs)
    train_step = make_train_step(loss_fn, optimizer)

    batch_size = cfg.THETA["batch_size"]
    n_iterations = cfg.THETA["n_iterations"]
    loss_history = np.zeros(n_iterations)

    print(f"training: {n_iterations} iterations, batch={batch_size}, w={w}")
    for i in range(n_iterations):
        key, idx_key, t0_key, *theta_keys = jax.random.split(key, batch_size + 3)
        idxs = jax.random.randint(idx_key, (batch_size,), 0, n_traj)
        t0s = jax.random.randint(t0_key, (batch_size,), 0, T - n_rollout + 1)
        theta_keys = jnp.stack(theta_keys)

        params, opt_state, loss = train_step(params, opt_state, theta_keys, idxs, t0s)
        loss_history[i] = float(loss)

        if i % 10 == 0 or i == n_iterations - 1:
            print(f"  iter {i:5d}  loss = {float(loss):.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump(params, f)
    np.save(LOSS_PATH, loss_history)
    print(f"saved {PARAMS_PATH}")
    print(f"saved {LOSS_PATH}")


if __name__ == "__main__":
    main()
