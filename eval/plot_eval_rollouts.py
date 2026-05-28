"""Visualize closed-loop eval rollouts: PD / pure / two-model on sampled DR plants.

This runs the closed-loop control law through MJX on perturbed plants, so it shows how each *trained* controller behaves end-to-end and where it fails. Controllers are built
exactly as in evaluate.py, so these rollouts match the scored metrics.

Writes:
    eval_rollouts_timeseries.png  joint angles vs reference, per plant, all controllers
    eval_rollouts_cartesian.png   strobe stick-figures of each controller's swing-up
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless WSL: render to file
import matplotlib.pyplot as plt
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
TIMESERIES_FIG_PATH = OUTPUT_DIR / "eval_rollouts_timeseries.png"
CARTESIAN_FIG_PATH = OUTPUT_DIR / "eval_rollouts_cartesian.png"

N_PLANTS = 3              # how many sampled DR plants to visualize
VIZ_SEED = cfg.EVAL_SEED  # same seed family as evaluate.py
N_STROBE = 14             # stick-figure snapshots per cartesian panel
L1 = L2 = 1.0             # link lengths from model.xml
NAMES = ["pd", "pure", "two_model"]
COLORS = {"pd": "tab:gray", "pure": "tab:orange", "two_model": "tab:blue"}


def forward_kinematics(q):
    """Joint angles (..., 2) -> (x1, z1, x2, z2): elbow and tip in the x-z plane."""
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    x1 = -L1 * np.sin(q1)
    z1 = -L1 * np.cos(q1)
    x2 = x1 - L2 * np.sin(q12)
    z2 = z1 - L2 * np.cos(q12)
    return x1, z1, x2, z2


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


def build_controllers(pure_params, theta_params, controller_params):
    """The three controller_fns, identical to evaluate.py."""
    pure_net = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
    theta_net = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
    ctrl_net = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)

    def pd_fn(*_):
        return jnp.zeros(cfg.NU)

    def pure_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
        net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
        return pure_net.apply(pure_params, net_in)

    def two_model_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
        theta_in = rollout.make_network_input(x_hist_full, u_hist)
        theta_hat = theta_net.apply(theta_params, theta_in)
        ctrl_in = rollout.make_network_input(
            x_hist_full, u_hist, x_ref_window, u_ref_window, theta_estimate=theta_hat)
        return ctrl_net.apply(controller_params, ctrl_in)

    return {"pd": pd_fn, "pure": pure_fn, "two_model": two_model_fn}


def make_rollout_fn(mjx_model_nominal, nominal_body_mass, controller_fn, T, w):
    """jit'd full-horizon rollout: (theta, x_ref_full, u_ref_full) -> (xs_full, vs)."""
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def run(theta, x_ref_full, u_ref_full):
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_init = x_ref_full[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_full[0], u_ref_full[0], w)
        xs, _us, vs, x_final = rollout.rollout(
            mjx_model, x_init, x_ref_full, u_ref_full,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, controller_fn, kp, kd, T)
        return jnp.concatenate([xs, x_final[None]], axis=0), vs

    return jax.jit(run)


def plot_timeseries(refs, rollouts, save_path, dt):
    """Per plant: q1 and q2 vs time, reference (dashed) + each controller."""
    t_x = np.arange(refs[0].shape[0]) * dt
    fig, axes = plt.subplots(N_PLANTS, 2, figsize=(13, 3.2 * N_PLANTS), squeeze=False)
    for p in range(N_PLANTS):
        for j in range(2):  # j=0 -> q1, j=1 -> q2
            ax = axes[p, j]
            ax.plot(t_x, refs[p][:, j], "k--", lw=1.5, label="reference")
            ax.axhline(refs[p][-1, j], color="0.6", ls=":", alpha=0.7)  # target
            for name in NAMES:
                ax.plot(t_x, rollouts[(name, p)][:, j], color=COLORS[name],
                        alpha=0.85, label=name)
            ax.set_ylabel(f"q{j + 1} (rad)")
            ax.set_title(f"plant {p}: q{j + 1}")
            ax.grid(True, alpha=0.3)
            if p == 0 and j == 0:
                ax.legend(fontsize=8, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_cartesian(refs, rollouts, save_path):
    """Grid [plant x controller] of strobe stick-figures; red = target pose."""
    fig, axes = plt.subplots(N_PLANTS, len(NAMES),
                             figsize=(4 * len(NAMES), 4 * N_PLANTS), squeeze=False)
    for p in range(N_PLANTS):
        T = refs[p].shape[0] - 1
        snaps = np.linspace(0, T, N_STROBE).astype(int)
        tx1, tz1, tx2, tz2 = forward_kinematics(refs[p][-1, :cfg.NQ])  # target pose
        for c, name in enumerate(NAMES):
            ax = axes[p, c]
            q = rollouts[(name, p)][:, :cfg.NQ]
            x1, z1, x2, z2 = forward_kinematics(q)
            ax.plot(x2, z2, color=COLORS[name], lw=1, alpha=0.5, label="tip path")
            for k, s in enumerate(snaps):
                a = 0.12 + 0.88 * (k / (len(snaps) - 1))
                ax.plot([0, x1[s], x2[s]], [0, z1[s], z2[s]],
                        "-o", color="0.2", alpha=a, ms=3, lw=1.5)
            ax.plot([0, tx1, tx2], [0, tz1, tz2], "-o", color="tab:red", lw=2, ms=5, label="target")
            ax.plot(0, 0, "ks", ms=6)
            ax.set_aspect("equal")
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.grid(True, alpha=0.3)
            if p == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f"plant {p}", fontsize=11)
    fig.suptitle("closed-loop swing-up (faint = start, dark = end)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    x_refs, u_refs = load_trajectories()
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    w = cfg.PURE["n_history"]

    controllers = build_controllers(
        load_pkl(PURE_PARAMS_PATH), load_pkl(THETA_PARAMS_PATH), load_pkl(CONTROLLER_PARAMS_PATH))
    mjx_model_nominal, nominal_body_mass = build_mjx_model()
    rollout_fns = {name: make_rollout_fn(mjx_model_nominal, nominal_body_mass, fn, T, w)
                   for name, fn in controllers.items()}

    # Sample plants from the eval seed family.
    theta_keys = jax.random.split(jax.random.PRNGKey(VIZ_SEED), N_PLANTS)
    thetas = [sample_theta(k, cfg.N_LINKS, cfg.DR_RANGES) for k in theta_keys]
    idxs = [p % n_traj for p in range(N_PLANTS)]

    refs = [np.asarray(x_refs[idxs[p]]) for p in range(N_PLANTS)]   # (T+1, nx), for plotting
    rollouts = {}
    print(f"rolling out {N_PLANTS} plants x {len(NAMES)} controllers (T={T}) ...")
    for p in range(N_PLANTS):
        idx = idxs[p]
        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]
        x_target = x_refs[idx, -1]
        msg = f"  plant {p} (traj {idx}, theta={np.round(np.asarray(thetas[p]), 2)})"
        for name in NAMES:
            xs_full, vs = rollout_fns[name](thetas[p], x_ref_full, u_ref_full)
            ep = float(losses.endpoint_error(xs_full[-1], x_target, nq))
            vrms = float(jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1))))
            rollouts[(name, p)] = np.asarray(xs_full)
            msg += f"\n      {name:10s} endpoint={ep:.2f}  |v|rms={vrms:.2f}"
        print(msg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeseries(refs, rollouts, TIMESERIES_FIG_PATH, cfg.TIMESTEP)
    plot_cartesian(refs, rollouts, CARTESIAN_FIG_PATH)


if __name__ == "__main__":
    main()
