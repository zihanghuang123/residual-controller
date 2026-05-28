"""Full hold-out eval, PD vs pure_rnn (no two-model).

Symmetric to evaluate_pure.py — same N_EVAL_PLANTS / EVAL_SEED protocol, same
output shape — but the residual controller is the GRU instead of the MLP, so
this loads pure_rnn_params.pkl and uses rollout_rnn (no history window).

PD shares the same rollout function: its h is a 1-element dummy threaded through
unchanged, since PD has no state. The PD physics is identical either way (no
history dependence), so the pd numbers match evaluate_pure.py's pd numbers.

Reports endpoint, tracking (RMS), and residual magnitude |v|rms over N_EVAL_PLANTS.
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
from lib.networks import GRUPureController, gru_initial_state

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
PURE_RNN_PARAMS_PATH = OUTPUT_DIR / "pure_rnn_params.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics_pure_rnn.npz"


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


def make_eval_fn(mjx_model_nominal, nominal_body_mass, controller_fn, h0, x_refs, u_refs):
    """Vmappable eval_fn(theta_key, idx) -> (endpoint, tracking_mse, vrms).

    Uses rollout_rnn for both pd and pure_rnn; PD's h is a dummy threaded through.
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
        x_ref_for_loss = x_refs[idx]      # (T+1, nx) — terminal-inclusive
        x_target = x_refs[idx, -1]
        x_init = x_ref_full[0]

        xs, _us, vs, x_final = rollout.rollout_rnn(
            mjx_model, x_init, x_ref_full, u_ref_full,
            h0, controller_fn, kp, kd, T)
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        endpoint = losses.endpoint_error(x_final, x_target, nq)
        tracking = losses.tracking_loss(xs_full, x_ref_for_loss, nq)
        vrms = jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1)))
        return endpoint, tracking, vrms

    return eval_fn


def make_pd_controller():
    """h is unused; we just pass it through."""
    def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
        return h, jnp.zeros(cfg.NU)
    return controller_fn


def make_pure_rnn_controller(pure_network, pure_params):
    def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
        x_step = rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref)
        return pure_network.apply(pure_params, h, x_step)
    return controller_fn


def summarize(endpoint, tracking, vrms, name):
    """One-row summary; tracking reported as RMS to match endpoint units."""
    ep = np.asarray(endpoint)
    rms_track = np.sqrt(np.asarray(tracking))
    vr = np.asarray(vrms)
    print(f"  {name:10s} "
          f" endpoint: mean={ep.mean():.4f}  med={np.median(ep):.4f}  "
          f"p90={np.percentile(ep, 90):.4f}  max={ep.max():.4f}"
          f"   tracking(rms): mean={rms_track.mean():.4f}  med={np.median(rms_track):.4f}"
          f"   |v|rms: mean={vr.mean():.3f}")


def main():
    print("loading trajectories ...")
    x_refs, u_refs = load_trajectories()
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    print(f"  {n_traj} converged trajectories of length T={T}  (hidden={cfg.PURE_RNN['hidden_sizes']})")

    print("loading pure_rnn params + building network ...")
    pure_params = load_pkl(PURE_RNN_PARAMS_PATH)
    pure_network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    mjx_model_nominal, nominal_body_mass = build_mjx_model()

    print(f"sampling {cfg.N_EVAL_PLANTS} eval plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), cfg.N_EVAL_PLANTS)
    idxs = jnp.arange(cfg.N_EVAL_PLANTS) % n_traj

    # PD has no state; rollout_rnn still needs *some* carry for h, so use a
    # 1-element dummy that the pd controller_fn passes through unchanged.
    pd_h0 = jnp.zeros(1)
    rnn_h0 = gru_initial_state(cfg.PURE_RNN["hidden_sizes"])

    controllers = {
        "pd": (make_pd_controller(), pd_h0),
        "pure_rnn": (make_pure_rnn_controller(pure_network, pure_params), rnn_h0),
    }

    results = {}
    print("evaluating ...")
    for name, (controller_fn, h0) in controllers.items():
        eval_fn = make_eval_fn(mjx_model_nominal, nominal_body_mass, controller_fn, h0, x_refs, u_refs)
        batched_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        endpoints, trackings, vrmss = batched_eval(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings), np.asarray(vrmss))
        summarize(endpoints, trackings, vrmss, name)

    ep_pd = results["pd"][0].mean()
    ep_pure_rnn = results["pure_rnn"][0].mean()
    print(f"\npure_rnn reduces mean endpoint by {100 * (1 - ep_pure_rnn / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_pure_rnn:.3f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        METRICS_PATH,
        endpoint_pd=results["pd"][0],
        tracking_pd=results["pd"][1],
        vrms_pd=results["pd"][2],
        endpoint_pure_rnn=results["pure_rnn"][0],
        tracking_pure_rnn=results["pure_rnn"][1],
        vrms_pure_rnn=results["pure_rnn"][2],
    )
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
