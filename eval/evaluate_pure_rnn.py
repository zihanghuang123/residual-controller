"""Hold-out eval, PD vs pure_rnn: metrics + endpoint box plot + per-traj tracking and cartesian plots."""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib import evaluation, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import GRUPureController, gru_initial_state
from eval.plot_trajectories import forward_kinematics, LINK_LENGTH

N_SHOW = 3       # trajectories to plot
N_STROBE = 14    # stick-figure snapshots per cartesian panel


def rollout_states(model, x_ref, u_ref, h0, controller_fn, kp, kd, T):
    """Closed-loop rollout -> (T+1, 2nq) state array (open-loop = PD controller with kp=kd=0)."""
    xs, _us, _vs, x_final = rollout.rollout_rnn(model, x_ref[0], x_ref, u_ref, h0, controller_fn, kp, kd, T)
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

    # --- aggregate metrics over held-out plants ---
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

    # --- endpoint box plot ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([ep_pd, ep_rnn], showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["pd", "pure_rnn"])
    ax.set_ylabel("endpoint error")
    ax.set_title(f"endpoint over {cfg.N_EVAL_PLANTS} held-out plants")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "eval_box.png", dpi=120)
    plt.close(fig)
    print(f"saved {cfg.OUTPUT_DIR / 'eval_box.png'}")

    # --- per-trajectory tracking + cartesian (same theta across the 3 controllers) ---
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), N_SHOW)
    t_axis = np.arange(T + 1) * cfg.TIMESTEP
    bound = LINK_LENGTH * cfg.N_LINKS + 0.3
    snaps = np.linspace(0, T, N_STROBE).astype(int)

    fig_t, axes_t = plt.subplots(N_SHOW, nq, figsize=(4 * nq, 2.5 * N_SHOW), squeeze=False)
    fig_c, axes_c = plt.subplots(1, N_SHOW, figsize=(4 * N_SHOW, 4.5), squeeze=False)

    pd_ctrl = lambda h, *a: pd_apply(None, h, *a)              # open-loop and pd differ only in gains
    rnn_ctrl = lambda h, *a: rnn_apply(pure_params, h, *a)

    for row in range(N_SHOW):
        idx = row
        theta = sample_theta(theta_keys[row], cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref = x_refs[idx, :T]
        u_ref = u_refs[idx]
        x_nom = np.asarray(x_refs[idx])

        ol = rollout_states(model, x_ref, u_ref, jnp.zeros(1), pd_ctrl, zero_gain, zero_gain, T)
        pdc = rollout_states(model, x_ref, u_ref, jnp.zeros(1), pd_ctrl, kp, kd, T)
        rnnc = rollout_states(model, x_ref, u_ref, h0, rnn_ctrl, kp, kd, T)

        for j in range(nq):
            ax = axes_t[row, j]
            ax.plot(t_axis, x_nom[:, j], "k", lw=1.5, label="reference")
            ax.plot(t_axis, ol[:, j], "tab:red", lw=1, label="u_nom")
            ax.plot(t_axis, pdc[:, j], "tab:blue", lw=1, label="u_nom + pd")
            ax.plot(t_axis, rnnc[:, j], "tab:green", lw=1, label="u_nom + pd + rnn")
            ax.set_title(f"traj {idx}, q{j + 1}")
            ax.grid(True, alpha=0.3)
            if row == 0 and j == 0:
                ax.legend(fontsize=8)

        xs_ref, zs_ref = forward_kinematics(x_nom[:, :nq])
        xs_rnn, zs_rnn = forward_kinematics(rnnc[:, :nq])
        ax = axes_c[0, row]
        ax.plot(xs_ref[:, -1], zs_ref[:, -1], "k", lw=1, alpha=0.5, label="reference tip")
        for k, s in enumerate(snaps):
            a = 0.12 + 0.88 * (k / (len(snaps) - 1))
            ax.plot(xs_rnn[s], zs_rnn[s], "-o", color="tab:green", alpha=a, ms=3, lw=1.5)
        ax.plot(xs_ref[-1], zs_ref[-1], "-o", color="k", lw=2, ms=4, label="reference final")
        ax.plot(0, 0, "ks", ms=6)
        ax.set_aspect("equal")
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_title(f"traj {idx}")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig_t.tight_layout()
    fig_t.savefig(cfg.OUTPUT_DIR / "eval_tracking.png", dpi=120)
    plt.close(fig_t)
    fig_c.suptitle("rnn rollout (faint = start, dark = end) vs reference tip")
    fig_c.tight_layout()
    fig_c.savefig(cfg.OUTPUT_DIR / "eval_cartesian.png", dpi=120)
    plt.close(fig_c)
    print(f"saved {cfg.OUTPUT_DIR / 'eval_tracking.png'}")
    print(f"saved {cfg.OUTPUT_DIR / 'eval_cartesian.png'}")


if __name__ == "__main__":
    main()
