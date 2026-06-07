"""Hold-out eval, PD vs pure (MLP): metrics + endpoint/tracking box plot + per-traj tracking plots."""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import numpy as np

from lib import evaluation, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPPureController

N_SHOW = 3   # trajectories to plot


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_pd_factory(cfg):
    def make_controller_fn(_theta):
        return lambda *_: jnp.zeros(cfg.NU)
    return make_controller_fn


def make_pure_factory(network, params):
    def make_controller_fn(_theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
            return network.apply(params, net_in)
        return controller_fn
    return make_controller_fn


def rollout_states(model, x_ref_full, u_ref_full, controller_fn, kp, kd, w, T):
    """Closed-loop rollout -> (T+1, 2nq) state array (open-loop = zero controller with kp=kd=0)."""
    x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = training.pad_history(x_ref_full[0], u_ref_full[0], w)
    xs, _us, _vs, x_final = rollout.rollout(
        model, x_ref_full[0], x_ref_full, u_ref_full,
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, controller_fn, kp, kd, T)
    return np.asarray(jnp.concatenate([xs, x_final[None]], axis=0))


def main():
    cfg = training.load_config()
    nq = cfg.NQ
    w = cfg.PURE["n_history"]
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    zero_gain = jnp.zeros(nq)

    print("loading trajectories + params ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    T = x_refs.shape[1] - 1
    pure_params = load_pkl(cfg.OUTPUT_DIR / "pure_params.pkl")
    pure_network = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    controllers = {
        "pd": make_pd_factory(cfg),
        "pure": make_pure_factory(pure_network, pure_params),
    }
    print(f"evaluating {cfg.N_EVAL_PLANTS} plants ...")
    results = evaluation.evaluate_residual_controllers(
        cfg, controllers, x_refs, u_refs, w, mjx_model_nominal, nominal_body_mass)
    ep_pd, ep_pure = results["pd"][0], results["pure"][0]
    print(f"\npure reduces mean endpoint by {100 * (1 - ep_pure.mean() / ep_pd.mean()):.1f}% vs pd "
          f"({ep_pd.mean():.3f} -> {ep_pure.mean():.3f})")
    evaluation.save_metrics(results, cfg.OUTPUT_DIR / "metrics_pure.npz")

    evaluation.plot_endpoint_tracking_box(results, "pure", cfg.N_EVAL_PLANTS, cfg.OUTPUT_DIR / "eval_box.png")

    # per-trajectory tracking (same theta across the 3 controllers)
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), N_SHOW)
    t_axis = np.arange(T + 1) * cfg.TIMESTEP
    zero_ctrl = lambda *_: jnp.zeros(cfg.NU)
    pure_ctrl = make_pure_factory(pure_network, pure_params)(None)

    rows = []
    for row in range(N_SHOW):
        theta = sample_theta(theta_keys[row], cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref_full = x_refs[row, :T]
        u_ref_full = u_refs[row]
        x_nom = np.asarray(x_refs[row])
        ol = rollout_states(model, x_ref_full, u_ref_full, zero_ctrl, zero_gain, zero_gain, w, T)
        pdc = rollout_states(model, x_ref_full, u_ref_full, zero_ctrl, kp, kd, w, T)
        pur = rollout_states(model, x_ref_full, u_ref_full, pure_ctrl, kp, kd, w, T)
        rows.append((row, [("reference", "k", 1.5, x_nom),
                           ("u_nom", "tab:red", 1.0, ol),
                           ("u_nom + pd", "tab:blue", 1.0, pdc),
                           ("u_nom + pd + mlp", "tab:green", 1.0, pur)]))
    evaluation.plot_tracking(rows, t_axis, nq, cfg.OUTPUT_DIR / "eval_tracking.png")


if __name__ == "__main__":
    main()
