"""Shared scaffolding for residual controller hold-out eval.

The make_controller_fn(theta) factory pattern keeps PD / pure / oracle / two-model uniform: pure ignores theta, oracle uses true theta, two-model computes theta_hat from history. The eval_fn samples theta per plant and constructs the rollout controller_fn from it.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lib import losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta


def make_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, x_refs, u_refs, w, make_controller_fn):
    """Vmappable eval_fn(theta_key, idx) -> (endpoint, tracking_mse, vrms).
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
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = training.pad_history(
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


def summarize(endpoint, tracking, vrms, name, width=10):
    """One-row eval summary; tracking reported as RMS to match endpoint units."""
    ep = np.asarray(endpoint)
    rms_track = np.sqrt(np.asarray(tracking))
    vr = np.asarray(vrms)
    print(f"  {name:{width}s} "
          f" endpoint: mean={ep.mean():.4f}  med={np.median(ep):.4f}  "
          f"p90={np.percentile(ep, 90):.4f}  max={ep.max():.4f}"
          f"   tracking(rms): mean={rms_track.mean():.4f}  med={np.median(rms_track):.4f}"
          f"   |v|rms: mean={vr.mean():.3f}")


def evaluate_residual_controllers(
    cfg, controllers, x_refs, u_refs, w,
    mjx_model_nominal, nominal_body_mass,
    name_width=10,
):
    """Run hold-out eval for a dict of {name: make_controller_fn} controllers.
    """
    n_traj = x_refs.shape[0]
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), cfg.N_EVAL_PLANTS)
    idxs = jnp.arange(cfg.N_EVAL_PLANTS) % n_traj

    results = {}
    for name, make_controller_fn in controllers.items():
        eval_fn = make_eval_fn(
            cfg, mjx_model_nominal, nominal_body_mass, x_refs, u_refs, w, make_controller_fn)
        batched_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        endpoints, trackings, vrmss = batched_eval(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings), np.asarray(vrmss))
        summarize(endpoints, trackings, vrmss, name, width=name_width)
    return results


def save_metrics(results, metrics_path):
    """Save {name: (endpoint, tracking, vrms)} to npz with conventional key names (endpoint_<name>, tracking_<name>, vrms_<name>)."""
    data = {}
    for name, (ep, tr, vr) in results.items():
        data[f"endpoint_{name}"] = ep
        data[f"tracking_{name}"] = tr
        data[f"vrms_{name}"] = vr
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(metrics_path, **data)
    print(f"saved {metrics_path}")


def eval_mlp_residual(cfg, controllers, traj_path, metrics_path, w, name_width=10):
    """End-to-end MLP residual controller eval.

    Caller pre-builds the `controllers` dict — `{name: make_controller_fn}`
    """
    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(traj_path)
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    print(f"  {n_traj} converged trajectories of length T={T}  (w={w})")

    print("building MJX model ...")
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    print(f"sampling {cfg.N_EVAL_PLANTS} eval plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    print("evaluating ...")
    results = evaluate_residual_controllers(
        cfg, controllers, x_refs, u_refs, w,
        mjx_model_nominal, nominal_body_mass,
        name_width=name_width,
    )

    save_metrics(results, metrics_path)
    return results
