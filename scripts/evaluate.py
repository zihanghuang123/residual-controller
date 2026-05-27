"""Hold-out evaluation: roll PD, pure, and two-model controllers over N_EVAL_PLANTS
plants drawn from the DR distribution under EVAL_SEED, report endpoint and tracking error statistics.

Endpoint error = ||x_final - x_target|| (RMS, with angle wrap).
Tracking error = mean over time of ||x_t - x_ref_t||^2 (MSE, with angle wrap).
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
from lib import losses, rollout
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPController, MLPPureController, MLPThetaEstimator

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
PURE_PARAMS_PATH = OUTPUT_DIR / "pure_params.pkl"
THETA_PARAMS_PATH = OUTPUT_DIR / "theta_params.pkl"
CONTROLLER_PARAMS_PATH = OUTPUT_DIR / "controller_params.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.npz"


def load_trajectories():
    """Load TO trajectories, keep only the converged ones."""
    data = np.load(TRAJ_PATH)
    mask = data["converged"].astype(bool)
    x_refs = jnp.asarray(data["x_refs"][mask])
    u_refs = jnp.asarray(data["u_refs"][mask])
    return x_refs, u_refs


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_mjx_model():
    """Load the MuJoCo model and push it to device as an mjx.Model."""
    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model.body_mass)
    return mjx_model, nominal_body_mass


def make_history_buffers(x_ref_t0, u_ref_t0, w):
    """Pad all four history buffers with the reference at the window start."""
    x_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_hist0 = jnp.tile(u_ref_t0, (w, 1))
    x_ref_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_ref_hist0 = jnp.tile(u_ref_t0, (w, 1))
    return x_hist0, u_hist0, x_ref_hist0, u_ref_hist0


def make_eval_fn(mjx_model_nominal, nominal_body_mass, controller_fn, x_refs, u_refs, w):
    """Return a vmappable eval_fn(theta_key, idx) -> (endpoint, tracking)."""
    T = x_refs.shape[1] - 1   # rollout length matches u_refs (T entries)
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_full = x_refs[idx, :T]              # (T, nx) — references aligned with rollout steps
        u_ref_full = u_refs[idx]                  # (T, nu)
        x_target = x_refs[idx, -1]                # (nx,) — TO final state = target

        x_init = x_ref_full[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_full[0], u_ref_full[0], w
        )

        xs, _us, _vs = rollout.rollout(
            mjx_model, x_init, x_ref_full, u_ref_full,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0,
            controller_fn, kp, kd, T,
        )

        endpoint = losses.endpoint_error(xs[-1], x_target, nq)
        tracking = losses.tracking_loss(xs, x_ref_full, nq)
        return endpoint, tracking

    return eval_fn


def make_pd_controller():
    """Zero residual — PD-only baseline."""
    def controller_fn(*_):
        return jnp.zeros(cfg.NU)
    return controller_fn


def make_pure_controller(pure_network, pure_params):
    """Pure MLP controller."""
    def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
        net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
        return pure_network.apply(pure_params, net_in)
    return controller_fn


def make_two_model_controller(theta_network, theta_params, controller_network, controller_params):
    """Two-model: frozen theta estimator + controller."""
    def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
        theta_input = rollout.make_network_input(x_hist_full, u_hist)
        theta_hat = theta_network.apply(theta_params, theta_input)
        controller_input = rollout.make_network_input(
            x_hist_full, u_hist, x_ref_window, u_ref_window,
            theta_estimate=theta_hat,
        )
        return controller_network.apply(controller_params, controller_input)
    return controller_fn


def summarize(endpoint, tracking, name):
    """Print a one-row summary line for one controller."""
    # tracking is MSE; report sqrt to put it in the same units as endpoint.
    rms_tracking = np.sqrt(np.asarray(tracking))
    ep = np.asarray(endpoint)
    print(f"  {name:12s} "
          f" endpoint: mean={ep.mean():.4f}  med={np.median(ep):.4f}  p90={np.percentile(ep, 90):.4f}  max={ep.max():.4f}"
          f"   tracking(rms): mean={rms_tracking.mean():.4f}  med={np.median(rms_tracking):.4f}  p90={np.percentile(rms_tracking, 90):.4f}")


def main():
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories()
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    print(f"  {n_traj} converged trajectories of length T={T}")

    print("loading params ...")
    pure_params = load_pkl(PURE_PARAMS_PATH)
    theta_params = load_pkl(THETA_PARAMS_PATH)
    controller_params = load_pkl(CONTROLLER_PARAMS_PATH)

    print("building networks ...")
    pure_network = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
    theta_network = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    controller_network = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print(f"sampling {cfg.N_EVAL_PLANTS} eval plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    key = jax.random.PRNGKey(cfg.EVAL_SEED)
    theta_keys = jax.random.split(key, cfg.N_EVAL_PLANTS)
    # Cycle through TO trajectories so each gets ~equal eval weight.
    idxs = jnp.arange(cfg.N_EVAL_PLANTS) % n_traj

    w = cfg.PURE["n_history"] 
    
    controllers = {
        "pd": make_pd_controller(),
        "pure": make_pure_controller(pure_network, pure_params),
        "two_model": make_two_model_controller(
            theta_network, theta_params, controller_network, controller_params
        ),
    }

    results = {}
    print("evaluating ...")
    for name, controller_fn in controllers.items():
        eval_fn = make_eval_fn(mjx_model_nominal, nominal_body_mass, controller_fn, x_refs, u_refs, w)
        batched_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        endpoints, trackings = batched_eval(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings))
        summarize(endpoints, trackings, name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        METRICS_PATH,
        endpoint_pd=results["pd"][0], 
        tracking_pd=results["pd"][1],
        endpoint_pure=results["pure"][0], 
        tracking_pure=results["pure"][1],
        endpoint_two_model=results["two_model"][0],
        tracking_two_model=results["two_model"][1],
    )
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
