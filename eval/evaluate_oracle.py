"""Full hold-out eval, PD vs oracle controller.

Same N_EVAL_PLANTS / EVAL_SEED protocol as evaluate_pure.py. The oracle controller is the MLPController (same arch as the two-model controller) but is fed the true theta sampled at rollout time, bypassing the estimator. Reports endpoint, tracking (RMS), and residual magnitude |v|rms.

Compare metrics_oracle.npz against metrics_pure.npz: if the oracle reduces endpoint vs pure, theta information is worth fighting for. If oracle ≈ pure, pure is already at the achievable ceiling.
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
from lib.networks import MLPController

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
ORACLE_PARAMS_PATH = OUTPUT_DIR / "oracle_params.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics_oracle.npz"


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


def make_eval_fn(mjx_model_nominal, nominal_body_mass, make_controller_fn, x_refs, u_refs, w):
    """Vmappable eval_fn(theta_key, idx) -> (endpoint, tracking_mse, vrms).

    make_controller_fn(theta) -> controller_fn — built per-plant so oracle can close over the sampled theta. PD ignores its argument.
    """
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]
        x_ref_for_loss = x_refs[idx]
        x_target = x_refs[idx, -1]

        x_init = x_ref_full[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_full[0], u_ref_full[0], w)

        controller_fn = make_controller_fn(theta)

        xs, _us, vs, x_final = rollout.rollout(
            mjx_model, x_init, x_ref_full, u_ref_full,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, controller_fn, kp, kd, T)
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        endpoint = losses.endpoint_error(x_final, x_target, nq)
        tracking = losses.tracking_loss(xs_full, x_ref_for_loss, nq)
        vrms = jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1)))
        return endpoint, tracking, vrms

    return eval_fn


def make_pd_controller_factory():
    def make_controller_fn(_theta):
        def controller_fn(*_):
            return jnp.zeros(cfg.NU)
        return controller_fn
    return make_controller_fn


def make_oracle_controller_factory(network, params):
    def make_controller_fn(theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            controller_input = rollout.make_network_input(
                x_hist_full, u_hist, x_ref_window, u_ref_window,
                theta_estimate=theta,
            )
            return network.apply(params, controller_input)
        return controller_fn
    return make_controller_fn


def summarize(endpoint, tracking, vrms, name):
    ep = np.asarray(endpoint)
    rms_track = np.sqrt(np.asarray(tracking))
    vr = np.asarray(vrms)
    print(f"  {name:6s} "
          f" endpoint: mean={ep.mean():.4f}  med={np.median(ep):.4f}  "
          f"p90={np.percentile(ep, 90):.4f}  max={ep.max():.4f}"
          f"   tracking(rms): mean={rms_track.mean():.4f}  med={np.median(rms_track):.4f}"
          f"   |v|rms: mean={vr.mean():.3f}")


def main():
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories()
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    w = cfg.ORACLE["n_history"]
    print(f"  {n_traj} converged trajectories of length T={T}  (w={w})")

    print("loading oracle params + building network ...")
    oracle_params = load_pkl(ORACLE_PARAMS_PATH)
    oracle_network = MLPController(hidden_sizes=cfg.ORACLE["hidden_sizes"], out_dim=cfg.NU)
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print(f"sampling {cfg.N_EVAL_PLANTS} eval plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), cfg.N_EVAL_PLANTS)
    idxs = jnp.arange(cfg.N_EVAL_PLANTS) % n_traj

    factories = {
        "pd": make_pd_controller_factory(),
        "oracle": make_oracle_controller_factory(oracle_network, oracle_params),
    }

    results = {}
    print("evaluating ...")
    for name, make_controller_fn in factories.items():
        eval_fn = make_eval_fn(mjx_model_nominal, nominal_body_mass, make_controller_fn, x_refs, u_refs, w)
        batched_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        endpoints, trackings, vrmss = batched_eval(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings), np.asarray(vrmss))
        summarize(endpoints, trackings, vrmss, name)

    ep_pd = results["pd"][0].mean()
    ep_oracle = results["oracle"][0].mean()
    print(f"\noracle reduces mean endpoint by {100 * (1 - ep_oracle / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_oracle:.3f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        METRICS_PATH,
        endpoint_pd=results["pd"][0], tracking_pd=results["pd"][1], vrms_pd=results["pd"][2],
        endpoint_oracle=results["oracle"][0], tracking_oracle=results["oracle"][1], vrms_oracle=results["oracle"][2],
    )
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
