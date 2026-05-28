"""Per-parameter estimator diagnostic: how well is each theta component identified?

The aggregate theta_loss averages 6 components with very different prior variances, so it hides which parameters are actually recoverable. This runs the trained estimator on freshly sampled plants and reports, per component, the RMSE and R^2 = 1 - MSE/Var.
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
from mujoco import mjx

from double_pendulum import config as cfg
from lib import rollout
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPThetaEstimator

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
THETA_PARAMS_PATH = OUTPUT_DIR / "theta_params.pkl"

N_PROBE = 2000  # plants to evaluate identifiability over


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trajectories():
    data = np.load(TRAJ_PATH)
    mask = data["converged"].astype(bool)
    return jnp.asarray(data["x_refs"][mask]), jnp.asarray(data["u_refs"][mask])


def build_mjx_model():
    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    return mjx_model, jnp.asarray(mjx_model.body_mass)


def make_history_buffers(x_ref_t0, u_ref_t0, w):
    x_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_hist0 = jnp.tile(u_ref_t0, (w, 1))
    x_ref_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_ref_hist0 = jnp.tile(u_ref_t0, (w, 1))
    return x_hist0, u_hist0, x_ref_hist0, u_ref_hist0


def make_predict_fn(mjx_model_nominal, nominal_body_mass, network, params, x_refs, u_refs, w):
    """Vmappable: (theta_key, idx, t0) -> (theta_true, theta_pred). Mirrors training."""
    n_rollout = w + 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def zero_controller(*_):
        return jnp.zeros(cfg.NU)

    def predict_fn(theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (n_rollout, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (n_rollout, cfg.NU))

        x_init = x_ref_window[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_window[0], u_ref_window[0], w)

        xs, us, _vs, _xf = rollout.rollout(
            mjx_model, x_init, x_ref_window, u_ref_window,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, zero_controller, kp, kd, n_rollout)

        net_in = rollout.make_network_input(xs, us[:w])
        theta_pred = network.apply(params, net_in)
        return theta, theta_pred

    return predict_fn


def main():
    x_refs, u_refs = load_trajectories()
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    w = cfg.THETA["n_history"]
    n_rollout = w + 1
    print(f"{n_traj} trajectories, T={T}, estimator w={w}, hidden={cfg.THETA['hidden_sizes']}")

    params = load_pkl(THETA_PARAMS_PATH)
    network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    key = jax.random.PRNGKey(cfg.EVAL_SEED)
    key, idx_key, t0_key = jax.random.split(key, 3)
    theta_keys = jax.random.split(key, N_PROBE)
    idxs = jax.random.randint(idx_key, (N_PROBE,), 0, n_traj)
    t0s = jax.random.randint(t0_key, (N_PROBE,), 0, T - n_rollout + 1)

    predict_fn = make_predict_fn(mjx_model_nominal, nominal_body_mass, network, params, x_refs, u_refs, w)
    batched = jax.jit(jax.vmap(predict_fn, in_axes=(0, 0, 0)))
    print(f"probing {N_PROBE} plants ...")
    theta_true, theta_pred = batched(theta_keys, idxs, t0s)
    theta_true = np.asarray(theta_true)
    theta_pred = np.asarray(theta_pred)

    err = theta_pred - theta_true
    mse = (err ** 2).mean(axis=0)
    var = theta_true.var(axis=0)
    rmse = np.sqrt(mse)
    r2 = 1.0 - mse / var

    labels = ([f"mass{i + 1}" for i in range(cfg.N_LINKS)] +
              [f"damp{i + 1}" for i in range(cfg.N_LINKS)] +
              [f"fric{i + 1}" for i in range(cfg.N_LINKS)])

    print(f"\naggregate MSE = {mse.mean():.5f}  (cross-check vs training loss)\n")
    print(f"  {'param':8s} {'true range':>14s} {'RMSE':>8s} {'R^2':>8s}")
    print("  " + "-" * 42)
    for j, name in enumerate(labels):
        lo, hi = theta_true[:, j].min(), theta_true[:, j].max()
        print(f"  {name:8s} {f'[{lo:.2f}, {hi:.2f}]':>14s} {rmse[j]:>8.4f} {r2[j]:>8.3f}")
    print("\n  R^2: 1=perfect, 0=no better than guessing the mean, <0=worse than the mean")


if __name__ == "__main__":
    main()
