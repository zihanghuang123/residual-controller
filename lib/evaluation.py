"""Shared scaffolding for residual controller hold-out eval.

The make_controller_fn(theta) factory pattern keeps PD / pure / oracle / two-model uniform: pure ignores theta, oracle uses true theta, two-model computes theta_hat from history. The eval_fn samples theta per plant and constructs the rollout controller_fn from it.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from lib import losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta


def _eval_plants(cfg, n_traj, n_eval):
    keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), n_eval)
    return keys, jnp.arange(n_eval) % n_traj


def _rollout_metrics(xs, vs, x_final, x_ref_full, nq):
    """Shared endpoint / tracking / |v|rms tail for both rollout paths."""
    xs_full = jnp.concatenate([xs, x_final[None]], axis=0)
    return (losses.endpoint_error(x_final, x_ref_full[-1], nq),
            losses.tracking_loss(xs_full, x_ref_full, nq),
            jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1))))


def _clamp_fraction(us, mjx_model):
    """Per-joint fraction of rollout steps with applied torque at a finite ctrl limit."""
    u_lo, u_hi = rollout._ctrl_limits(mjx_model)
    tol = 1e-5
    at = (us >= u_hi - tol) | (us <= u_lo + tol)
    return jnp.mean(at.astype(jnp.float32), axis=0)


def _update_best(metric, state, params, path):
    """Save params to path when metric beats state['best']; returns True if it was a new best."""
    if metric >= state["best"]:
        return False
    state["best"] = metric
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)
    return True


def make_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, x_refs, u_refs, w, make_controller_fn):
    """Vmappable eval_fn(theta_key, idx) -> (endpoint, tracking_mse, vrms)."""
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]

        x_init = x_ref_full[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = training.pad_history(
            x_ref_full[0], u_ref_full[0], w)

        controller_fn = make_controller_fn(theta)

        xs, us, vs, x_final = rollout.rollout(
            mjx_model, x_init, x_ref_full, u_ref_full,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, controller_fn, kp, kd, T)
        return (*_rollout_metrics(xs, vs, x_final, x_refs[idx], nq), _clamp_fraction(us, mjx_model))

    return eval_fn


def summarize(endpoint, tracking, vrms, clamp, name, width=10):
    """One-row eval summary; tracking reported as RMS. clamp: (n_plants, nq) per-joint hit fractions."""
    ep = np.asarray(endpoint)
    rms_track = np.sqrt(np.asarray(tracking))
    vr = np.asarray(vrms)
    cl = np.asarray(clamp)
    print(f"  {name:{width}s} "
          f" endpoint: mean={ep.mean():.4f}  med={np.median(ep):.4f}  "
          f"p90={np.percentile(ep, 90):.4f}  max={ep.max():.4f}"
          f"   tracking(rms): mean={rms_track.mean():.4f}  med={np.median(rms_track):.4f}"
          f"   |v|rms: mean={vr.mean():.3f}"
          f"   clamp-hit: mean={cl.mean():.3f} per-joint={np.round(cl.mean(axis=0), 2)}")


def evaluate_residual_controllers(
    cfg, controllers, x_refs, u_refs, w,
    mjx_model_nominal, nominal_body_mass,
    name_width=10,
):
    """Run hold-out eval for a dict of {name: make_controller_fn} controllers."""
    n_traj = x_refs.shape[0]
    theta_keys, idxs = _eval_plants(cfg, n_traj, cfg.N_EVAL_PLANTS)

    results = {}
    for name, make_controller_fn in controllers.items():
        eval_fn = make_eval_fn(
            cfg, mjx_model_nominal, nominal_body_mass, x_refs, u_refs, w, make_controller_fn)
        batched_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        endpoints, trackings, vrmss, clamps = batched_eval(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings), np.asarray(vrmss), np.asarray(clamps))
        summarize(endpoints, trackings, vrmss, clamps, name, width=name_width)
    return results


def save_metrics(results, metrics_path):
    """Save {name: (endpoint, tracking, vrms)} to npz with conventional key names (endpoint_<name>, tracking_<name>, vrms_<name>)."""
    data = {}
    for name, (ep, tr, vr, cl) in results.items():
        data[f"endpoint_{name}"] = ep
        data[f"tracking_{name}"] = tr
        data[f"vrms_{name}"] = vr
        data[f"clamp_{name}"] = cl
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(metrics_path, **data)
    print(f"saved {metrics_path}")


def eval_mlp_residual(cfg, controllers, traj_path, metrics_path, w, name_width=10):
    """End-to-end MLP residual controller eval. Caller pre-builds {name: make_controller_fn}."""
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

    if metrics_path is not None:
        save_metrics(results, metrics_path)
    return results


def make_eval_callback(cfg, make_controllers, target_name, w,
                       csv_path, best_params_path, traj_path=None,
                       best_opt_state_path=None):
    """Build a periodic eval callback for the training loop.
    """
    if traj_path is None:
        traj_path = cfg.OUTPUT_DIR / "trajectories.npz"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("iter,endpoint_pd,endpoint_target,tracking_rms_target,vrms_target,is_best\n")

    state = {"best": float("inf")}

    def callback(params, opt_state, iteration):
        controllers = make_controllers(params)
        results = eval_mlp_residual(
            cfg, controllers,
            traj_path=traj_path, metrics_path=None,
            w=w,
        )
        ep_target = float(results[target_name][0].mean())
        ep_pd = float(results["pd"][0].mean()) if "pd" in results else float("nan")
        tr_rms = float(np.sqrt(results[target_name][1]).mean())
        vr_mean = float(results[target_name][2].mean())

        is_best = _update_best(ep_target, state, params, best_params_path)
        if is_best and best_opt_state_path is not None:
            with open(best_opt_state_path, "wb") as f:
                pickle.dump(opt_state, f)

        with open(csv_path, "a") as f:
            f.write(f"{iteration},{ep_pd:.6f},{ep_target:.6f},"
                    f"{tr_rms:.6f},{vr_mean:.6f},{int(is_best)}\n")

        reduction = 100 * (1 - ep_target / ep_pd) if ep_pd > 0 else float("nan")
        suffix = "  *** BEST ***" if is_best else ""
        print(f"  [eval iter {iteration:5d}] {target_name}: endpoint={ep_target:.4f}  "
              f"vs pd {ep_pd:.4f}  ({reduction:.1f}% reduction){suffix}")

    return callback


# --- RNN (GRU) residual eval: rollout_rnn, no history window ---

def rnn_controller_apply(network):
    """controller_apply(params, h, x_curr, u_prev, x_ref, u_ref) -> (h, v) for rollout_rnn."""
    def apply(params, h, x_curr, u_prev, x_ref, u_ref):
        return network.apply(params, h, rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref))
    return apply


def pd_controller_apply(nu):
    """PD baseline: v=0, hidden state passed through unchanged."""
    def apply(params, h, x_curr, u_prev, x_ref, u_ref):
        return h, jnp.zeros(nu)
    return apply


def make_rnn_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, controller_apply, h0, x_refs, u_refs):
    """Vmappable eval_fn(params, theta_key, idx) -> (endpoint, tracking, vrms). params is an arg so it compiles once."""
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(params, theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]

        def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
            return controller_apply(params, h, x_curr, u_prev, x_ref, u_ref)

        xs, us, vs, x_final = rollout.rollout_rnn(
            mjx_model, x_ref_full[0], x_ref_full, u_ref_full, h0, controller_fn, kp, kd, T)
        return (*_rollout_metrics(xs, vs, x_final, x_refs[idx], nq), _clamp_fraction(us, mjx_model))

    return eval_fn


def evaluate_rnn_controllers(cfg, controllers, x_refs, u_refs, mjx_model_nominal, nominal_body_mass, name_width=10):
    """Hold-out eval for {name: (controller_apply, params, h0)} via rollout_rnn."""
    n_traj = x_refs.shape[0]
    theta_keys, idxs = _eval_plants(cfg, n_traj, cfg.N_EVAL_PLANTS)
    results = {}
    for name, (controller_apply, params, h0) in controllers.items():
        eval_fn = make_rnn_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, controller_apply, h0, x_refs, u_refs)
        ep, tr, vr, cl = jax.jit(jax.vmap(eval_fn, in_axes=(None, 0, 0)))(params, theta_keys, idxs)
        results[name] = (np.asarray(ep), np.asarray(tr), np.asarray(vr), np.asarray(cl))
        summarize(ep, tr, vr, cl, name, width=name_width)
    return results


def make_rnn_eval_callback(cfg, network, h0, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
                           csv_path, best_params_path, n_eval=64, best_opt_state_path=None):
    """Periodic closed-loop eval for the GRU trainer: logs CSV, saves best-endpoint params (+ opt_state). PD baseline computed once."""
    theta_keys, idxs = _eval_plants(cfg, x_refs.shape[0], n_eval)

    rnn_eval = jax.jit(jax.vmap(make_rnn_eval_fn(
        cfg, mjx_model_nominal, nominal_body_mass, rnn_controller_apply(network), h0, x_refs, u_refs),
        in_axes=(None, 0, 0)))
    pd_eval = jax.jit(jax.vmap(make_rnn_eval_fn(
        cfg, mjx_model_nominal, nominal_body_mass, pd_controller_apply(cfg.NU), jnp.zeros(1), x_refs, u_refs),
        in_axes=(None, 0, 0)))
    ep_pd = float(pd_eval(jnp.zeros(1), theta_keys, idxs)[0].mean())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("iter,endpoint_pd,endpoint_rnn,tracking_rms_rnn,vrms_rnn,clamp_rnn,is_best\n")
    state = {"best": float("inf")}

    def callback(params, iteration, opt_state=None):
        ep, tr, vr, cl = rnn_eval(params, theta_keys, idxs)
        ep_m, tr_rms, vr_m, cl_m = float(ep.mean()), float(np.sqrt(tr).mean()), float(vr.mean()), float(cl.mean())
        is_best = _update_best(ep_m, state, params, best_params_path)
        if is_best and opt_state is not None and best_opt_state_path is not None:
            with open(best_opt_state_path, "wb") as f:
                pickle.dump(opt_state, f)
        with open(csv_path, "a") as f:
            f.write(f"{iteration},{ep_pd:.6f},{ep_m:.6f},{tr_rms:.6f},{vr_m:.6f},{cl_m:.6f},{int(is_best)}\n")
        reduction = 100 * (1 - ep_m / ep_pd) if ep_pd > 0 else float("nan")
        suffix = "  *** BEST ***" if is_best else ""
        print(f"  [eval iter {iteration:5d}] pure_rnn endpoint={ep_m:.4f} vs pd {ep_pd:.4f} "
              f"({reduction:.1f}% reduction){suffix}")

    return callback


# --- shared eval plots (evaluate_pure.py / evaluate_pure_rnn.py) ---

def plot_endpoint_tracking_box(results, target_name, n_plants, out_path):
    """Two-panel box plot: endpoint error and tracking rms, pd vs target over held-out plants."""
    import matplotlib.pyplot as plt
    ep = [results["pd"][0], results[target_name][0]]
    tr = [np.sqrt(results["pd"][1]), np.sqrt(results[target_name][1])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, data, ylab in zip(axes, (ep, tr), ("endpoint error", "tracking rms")):
        ax.boxplot(data, showfliers=False)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["pd", target_name])
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"pd vs {target_name} over {n_plants} held-out plants")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_tracking(rows, t_axis, nq, out_path):
    """Per-trajectory joint tracking. rows: [(traj_id, [(label, color, lw, states), ...]), ...]."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), nq, figsize=(4 * nq, 2.5 * len(rows)), squeeze=False)
    for r, (traj_id, lines) in enumerate(rows):
        for j in range(nq):
            ax = axes[r, j]
            for label, color, lw, states in lines:
                ax.plot(t_axis, states[:, j], color=color, lw=lw, label=label)
            ax.set_title(f"traj {traj_id}, q{j + 1}")
            ax.grid(True, alpha=0.3)
            if r == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")
