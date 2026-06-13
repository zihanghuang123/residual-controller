"""Hold-out eval, PD vs pure_rnn: metrics + endpoint/tracking box plot + per-traj tracking plots."""

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
from lib.networks import GRUPureController, gru_initial_state

N_SHOW = 3   # trajectories to plot


def rollout_states(model, x_ref, u_ref, h0, controller_fn, kp, kd, T, preview=None):
    """Closed-loop rollout -> (T+1, 2nq) state array (open-loop = PD controller with kp=kd=0)."""
    xs, _us, _vs, x_final = rollout.rollout_rnn(model, x_ref[0], x_ref, u_ref, h0, controller_fn, kp, kd, T, preview=preview)
    return np.asarray(jnp.concatenate([xs, x_final[None]], axis=0))


def main():
    cfg = training.load_config()
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    zero_gain = jnp.zeros(nq)

    print("loading trajectories + params ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    T = x_refs.shape[1] - 1
    with open(cfg.OUTPUT_DIR / "pure_rnn_params.pkl", "rb") as f:
        pure_params = pickle.load(f)
    network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    h0 = gru_initial_state(cfg.PURE_RNN["hidden_sizes"])
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    pd_apply = evaluation.pd_controller_apply(cfg.NU)
    rnn_apply = evaluation.rnn_controller_apply(network)

    controllers = {
        "pd": (pd_apply, jnp.zeros(1), jnp.zeros(1)),
        "pure_rnn": (rnn_apply, pure_params, h0),
    }
    print(f"evaluating {cfg.N_EVAL_PLANTS} plants ...")
    results = evaluation.evaluate_rnn_controllers(
        cfg, controllers, x_refs, u_refs, mjx_model_nominal, nominal_body_mass)
    ep_pd, ep_rnn = results["pd"][0], results["pure_rnn"][0]
    print(f"\npure_rnn reduces mean endpoint by {100 * (1 - ep_rnn.mean() / ep_pd.mean()):.1f}% vs pd "
          f"({ep_pd.mean():.3f} -> {ep_rnn.mean():.3f})")
    evaluation.save_metrics(results, cfg.OUTPUT_DIR / "metrics_pure_rnn.npz")

    evaluation.plot_endpoint_tracking_box(results, "pure_rnn", cfg.N_EVAL_PLANTS, cfg.OUTPUT_DIR / "eval_box.png")

    # per-trajectory tracking (same theta across the 3 controllers)
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), N_SHOW)
    t_axis = np.arange(T + 1) * cfg.TIMESTEP
    pd_ctrl = lambda h, *a: pd_apply(None, h, *a)              # open-loop and pd differ only in gains
    rnn_ctrl = lambda h, *a: rnn_apply(pure_params, h, *a)
    n_points = cfg.PURE_RNN.get("lookahead_points", 0)
    stride = cfg.PURE_RNN.get("lookahead_stride", 1)

    rows = []
    for row in range(N_SHOW):
        theta = sample_theta(theta_keys[row], cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref = x_refs[row, :T]
        u_ref = u_refs[row]
        x_nom = np.asarray(x_refs[row])
        preview = None if n_points == 0 else rollout.build_preview_window(
            jnp.asarray(x_refs[row]), jnp.asarray(u_refs[row]), 0, T, n_points, stride)
        ol = rollout_states(model, x_ref, u_ref, jnp.zeros(1), pd_ctrl, zero_gain, zero_gain, T)
        pdc = rollout_states(model, x_ref, u_ref, jnp.zeros(1), pd_ctrl, kp, kd, T)
        rnnc = rollout_states(model, x_ref, u_ref, h0, rnn_ctrl, kp, kd, T, preview=preview)
        rows.append((row, [("reference", "k", 1.5, x_nom),
                           ("u_nom", "tab:red", 1.0, ol),
                           ("u_nom + pd", "tab:blue", 1.0, pdc),
                           ("u_nom + pd + rnn", "tab:green", 1.0, rnnc)]))
    evaluation.plot_tracking(rows, t_axis, nq, cfg.OUTPUT_DIR / "eval_tracking.png")


if __name__ == "__main__":
    main()
