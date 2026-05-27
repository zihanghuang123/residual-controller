"""Train the controller with frozen theta estimator (Stage 2 of two-model approach).

Loads the theta estimator params from Stage 1 and trains the controller end-to-end via BPTT through MJX under DR. At each rollout step the theta estimator predicts theta_hat from the current (x, u) history, and the controller sees theta_hat as an
extra input alongside history + reference. Theta params are captured in the closure (not function arguments) so jax.grad differentiates only w.r.t. controller params.
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
from lib.networks import MLPController, MLPThetaEstimator

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
THETA_PARAMS_PATH = OUTPUT_DIR / "theta_params.pkl"
PARAMS_PATH = OUTPUT_DIR / "controller_params.pkl"
LOSS_PATH = OUTPUT_DIR / "controller_loss_history.npy"


def load_trajectories():
    """Load TO trajectories, keep only the converged ones."""
    data = np.load(TRAJ_PATH)
    mask = data["converged"].astype(bool)
    x_refs = jnp.asarray(data["x_refs"][mask])
    u_refs = jnp.asarray(data["u_refs"][mask])
    return x_refs, u_refs


def load_theta_params():
    """Load the frozen theta estimator params from Stage 1."""
    with open(THETA_PARAMS_PATH, "rb") as f:
        return pickle.load(f)


def build_mjx_model():
    """Load the MuJoCo model and push it to device as an mjx.Model."""
    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model.body_mass)
    return mjx_model, nominal_body_mass


def init_theta_network():
    """Build the theta estimator module. Params are loaded separately from disk."""
    return MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)


def init_controller_network(key):
    """Build the controller and initialize its params from a dummy input."""
    network = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)
    w = cfg.CONTROLLER["n_history"]
    nx = 2 * cfg.NQ
    # Input: x_hist_full (w+1, nx) + u_hist (w, nu) + x_ref_window (w+1, nx) + u_ref_window (w+1, nu) + theta_estimate.
    in_dim = (w + 1) * nx + w * cfg.NU + (w + 1) * nx + (w + 1) * cfg.NU + cfg.THETA_DIM
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


def make_loss_fn(mjx_model_nominal, nominal_body_mass,
                 theta_network, theta_params,
                 controller_network, x_refs, u_refs):
    """Build the single-rollout loss fn. theta_params is captured (frozen)."""
    w = cfg.CONTROLLER["n_history"]
    H = cfg.CONTROLLER["n_rollout"]
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def loss_fn(controller_params, theta_key, idx, t0):
        # Sample theta and perturb the plant.
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (H, cfg.NU))

        x_init = x_ref_window[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_window[0], u_ref_window[0], w
        )

        def controller_fn(x_hist_full, u_hist, x_ref_window_w, u_ref_window_w):
            # Estimator: history-only input (refs carry no theta info).
            theta_input = rollout.make_network_input(x_hist_full, u_hist)
            theta_hat = theta_network.apply(theta_params, theta_input)
            # Controller: everything + theta estimate.
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window_w, u_ref_window_w,
                theta_estimate=theta_hat,
            )
            return controller_network.apply(controller_params, controller_input)

        xs, _us, vs = rollout.rollout(
            mjx_model, x_init, x_ref_window, u_ref_window,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0,
            controller_fn, kp, kd, H,
        )

        return losses.tracking_loss(xs, x_ref_window, nq) + cfg.CONTROLLER["alpha_reg"] * losses.reg_loss(vs)

    return loss_fn


def make_train_step(loss_fn, optimizer):
    """Vmap the loss over a batch, take grads, apply the Adam step. Returns a jit'd fn."""
    batched_loss = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))

    def mean_loss(controller_params, theta_keys, idxs, t0s):
        return jnp.mean(batched_loss(controller_params, theta_keys, idxs, t0s))

    grad_fn = jax.value_and_grad(mean_loss)

    @jax.jit
    def train_step(controller_params, opt_state, theta_keys, idxs, t0s):
        loss, grads = grad_fn(controller_params, theta_keys, idxs, t0s)
        updates, opt_state = optimizer.update(grads, opt_state, controller_params)
        controller_params = optax.apply_updates(controller_params, updates)
        return controller_params, opt_state, loss

    return train_step


def main():
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories()
    n_traj, T_plus_1, _ = x_refs.shape
    T = T_plus_1 - 1
    H = cfg.CONTROLLER["n_rollout"]
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("loading frozen theta estimator ...")
    theta_params = load_theta_params()
    theta_network = init_theta_network()

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print("initializing controller network ...")
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    controller_network, controller_params = init_controller_network(init_key)

    print("building optimizer ...")
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.CONTROLLER["grad_clip_norm"]),
        optax.adam(cfg.CONTROLLER["lr"]),
    )
    opt_state = optimizer.init(controller_params)

    loss_fn = make_loss_fn(
        mjx_model_nominal, nominal_body_mass,
        theta_network, theta_params,
        controller_network, x_refs, u_refs,
    )
    train_step = make_train_step(loss_fn, optimizer)

    batch_size = cfg.CONTROLLER["batch_size"]
    n_iterations = cfg.CONTROLLER["n_iterations"]
    loss_history = np.zeros(n_iterations)

    print(f"training: {n_iterations} iterations, batch={batch_size}, H={H}, w={cfg.CONTROLLER['n_history']}")
    for i in range(n_iterations):
        key, idx_key, t0_key, *theta_keys = jax.random.split(key, batch_size + 3)
        idxs = jax.random.randint(idx_key, (batch_size,), 0, n_traj)
        t0s = jax.random.randint(t0_key, (batch_size,), 0, T - H + 1)
        theta_keys = jnp.stack(theta_keys)

        controller_params, opt_state, loss = train_step(
            controller_params, opt_state, theta_keys, idxs, t0s
        )
        loss_history[i] = float(loss)

        if i % 10 == 0 or i == n_iterations - 1:
            print(f"  iter {i:5d}  loss = {float(loss):.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "wb") as f:
        pickle.dump(controller_params, f)
    np.save(LOSS_PATH, loss_history)
    print(f"saved {PARAMS_PATH}")
    print(f"saved {LOSS_PATH}")


if __name__ == "__main__":
    main()
