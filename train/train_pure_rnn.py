"""Train pure RNN (GRU) residual controller via BPTT through MJX under DR.

Random (trajectory, t0, plant) per iteration, BPTT through n_rollout MJX steps, tracking + control regularization
loss.

The hidden state is the history, so there's no n_history knob and no padded history buffers.
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
from lib.networks import GRUPureController, gru_initial_state

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
PARAMS_PATH = OUTPUT_DIR / "pure_rnn_params.pkl"
LOSS_PATH = OUTPUT_DIR / "pure_rnn_loss_history.npy"


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
    """Build the GRU controller and initialize params from a dummy (h, x) pair."""
    network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    nx = 2 * cfg.NQ
    # Per-step input: [x_curr, u_prev, x_ref, u_ref].
    step_in_dim = nx + cfg.NU + nx + cfg.NU
    dummy_h = gru_initial_state(cfg.PURE_RNN["hidden_sizes"])
    dummy_x = jnp.zeros(step_in_dim)
    params = network.init(key, dummy_h, dummy_x)
    return network, params


def make_loss_fn(mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs):
    """Build the single-rollout loss fn closed over fixed data."""
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

        x_init = x_ref_window[0]
        h0 = gru_initial_state(hidden_sizes)

        def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
            x_step = rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref)
            return network.apply(params, h, x_step)

        xs, _us, vs, x_final = rollout.rollout_rnn(
            mjx_model, x_init, x_ref_window, u_ref_window,
            h0, controller_fn, kp, kd, H,
        )
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        return losses.tracking_loss(xs_full, x_ref_for_loss, nq) + cfg.PURE_RNN["alpha_reg"] * losses.reg_loss(vs)

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
    H = cfg.PURE_RNN["n_rollout"]

    print(f"  {n_traj} converged trajectories of length T={T}")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print("initializing network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    network, params = init_network(init_key)

    print("building optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.PURE_RNN["grad_clip_norm"]),
        optax.adam(cfg.PURE_RNN["lr"]),
    )
    opt_state = optimizer.init(params)

    loss_fn = make_loss_fn(mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs)
    train_step = make_train_step(loss_fn, optimizer)

    batch_size = cfg.PURE_RNN["batch_size"]
    n_iterations = cfg.PURE_RNN["n_iterations"]
    loss_history = np.zeros(n_iterations)

    print(f"training: {n_iterations} iterations, batch={batch_size}, H={H}, hidden={cfg.PURE_RNN['hidden_sizes']}")
    for i in range(n_iterations):
        key, idx_key, t0_key, *theta_keys = jax.random.split(key, batch_size + 3)
        idxs = jax.random.randint(idx_key, (batch_size,), 0, n_traj)
        t0s = jax.random.randint(t0_key, (batch_size,), 0, T - H + 1)
        theta_keys = jnp.stack(theta_keys)

        params, opt_state, loss = train_step(params, opt_state, theta_keys, idxs, t0s)
        loss_history[i] = float(loss)

        if i % 50 == 0 or i == n_iterations - 1:
            print(f"  iter {i:5d}  loss = {float(loss):.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump(params, f)
    np.save(LOSS_PATH, loss_history)
    print(f"saved {PARAMS_PATH}")
    print(f"saved {LOSS_PATH}")


if __name__ == "__main__":
    main()
