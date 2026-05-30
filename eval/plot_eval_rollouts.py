"""Visualize closed-loop eval rollouts on sampled DR plants.

Runs the closed-loop control law through MJX on perturbed plants for whichever controllers are available (params files present in cfg.OUTPUT_DIR): pd, pure, oracle, two_model. Skips silently if a controller's params don't exist.

Writes:
    eval_rollouts_timeseries.png  joint angles vs reference, per plant, all available controllers
    eval_rollouts_cartesian.png   strobe stick-figures of each controller's swing-up

Works for any N-link pendulum.
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

from lib import losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import MLPController, MLPPureController, MLPThetaEstimator

LINK_LENGTH = 1.0    # all current plant XMLs use unit-length links
N_PLANTS = 3         # how many sampled DR plants to visualize
N_STROBE = 14        # stick-figure snapshots per cartesian panel

COLORS = {
    "pd": "tab:gray",
    "pure": "tab:orange",
    "oracle": "tab:green",
    "two_model": "tab:blue",
}


def forward_kinematics(q, link_length=LINK_LENGTH):
    """q: (..., N_LINKS) -> (xs, zs) each of shape (..., N_LINKS+1), including base (0, 0)."""
    cumq = np.cumsum(q, axis=-1)
    out_shape = q.shape[:-1] + (q.shape[-1] + 1,)
    xs = np.zeros(out_shape)
    zs = np.zeros(out_shape)
    for i in range(q.shape[-1]):
        xs[..., i + 1] = xs[..., i] - link_length * np.sin(cumq[..., i])
        zs[..., i + 1] = zs[..., i] - link_length * np.cos(cumq[..., i])
    return xs, zs


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trajectories(traj_path):
    data = np.load(traj_path)
    mask = data["converged"].astype(bool)
    return jnp.asarray(data["x_refs"][mask]), jnp.asarray(data["u_refs"][mask])


def build_mjx_model(model_path):
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mjx_model = mjx.put_model(mj_model)
    return mjx_model, jnp.asarray(mjx_model.body_mass)


def make_history_buffers(x_ref_t0, u_ref_t0, w):
    x_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_hist0 = jnp.tile(u_ref_t0, (w, 1))
    x_ref_hist0 = jnp.tile(x_ref_t0, (w, 1))
    u_ref_hist0 = jnp.tile(u_ref_t0, (w, 1))
    return x_hist0, u_hist0, x_ref_hist0, u_ref_hist0


def build_factories(cfg, output_dir):
    """Build {name: make_controller_fn(theta) -> 4-arg controller_fn} for all available controllers.

    Always includes PD. Adds pure / oracle / two_model only if their params files exist in output_dir.
    """
    def pd_factory(_theta):
        return lambda *_: jnp.zeros(cfg.NU)

    factories = {"pd": pd_factory}

    pure_path = output_dir / "pure_params.pkl"
    if pure_path.exists():
        net = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)
        params = load_pkl(pure_path)
        def pure_factory(_theta):
            def fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
                net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
                return net.apply(params, net_in)
            return fn
        factories["pure"] = pure_factory
        print(f"  found pure_params")

    oracle_path = output_dir / "oracle_params.pkl"
    if oracle_path.exists():
        net = MLPController(hidden_sizes=cfg.ORACLE["hidden_sizes"], out_dim=cfg.NU)
        params = load_pkl(oracle_path)
        def oracle_factory(theta):
            def fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
                inp = rollout.make_network_input(
                    x_hist_full, u_hist, x_ref_window, u_ref_window,
                    theta_estimate=theta)
                return net.apply(params, inp)
            return fn
        factories["oracle"] = oracle_factory
        print(f"  found oracle_params")

    theta_path = output_dir / "theta_params.pkl"
    controller_path = output_dir / "controller_params.pkl"
    if theta_path.exists() and controller_path.exists():
        theta_net = MLPThetaEstimator(hidden_sizes=cfg.THETA["hidden_sizes"], theta_dim=cfg.THETA_DIM)
        theta_params = load_pkl(theta_path)
        ctrl_net = MLPController(hidden_sizes=cfg.CONTROLLER["hidden_sizes"], out_dim=cfg.NU)
        ctrl_params = load_pkl(controller_path)
        def two_model_factory(_theta):
            def fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
                theta_in = rollout.make_network_input(x_hist_full, u_hist)
                theta_hat = theta_net.apply(theta_params, theta_in)
                ctrl_in = rollout.make_network_input(
                    x_hist_full, u_hist, x_ref_window, u_ref_window, theta_estimate=theta_hat)
                return ctrl_net.apply(ctrl_params, ctrl_in)
            return fn
        factories["two_model"] = two_model_factory
        print(f"  found theta + controller_params (two_model)")

    return factories


def make_rollout_fn(cfg, mjx_model_nominal, nominal_body_mass, make_controller_fn, T, w):
    """jit'd full-horizon rollout: (theta, x_ref_full, u_ref_full) -> (xs_full, vs)."""
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    n_links = cfg.N_LINKS

    def run(theta, x_ref_full, u_ref_full):
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, n_links)
        x_init = x_ref_full[0]
        x_hist0, u_hist0, x_ref_hist0, u_ref_hist0 = make_history_buffers(
            x_ref_full[0], u_ref_full[0], w)
        controller_fn = make_controller_fn(theta)
        xs, _us, vs, x_final = rollout.rollout(
            mjx_model, x_init, x_ref_full, u_ref_full,
            x_hist0, u_hist0, x_ref_hist0, u_ref_hist0, controller_fn, kp, kd, T)
        return jnp.concatenate([xs, x_final[None]], axis=0), vs

    return jax.jit(run)


def plot_timeseries(cfg, names, refs, rollouts, save_path):
    """Per plant: each joint angle vs time, reference (dashed) + each controller."""
    nq = cfg.NQ
    dt = cfg.TIMESTEP
    t_x = np.arange(refs[0].shape[0]) * dt

    fig, axes = plt.subplots(N_PLANTS, nq, figsize=(5 * nq, 3.2 * N_PLANTS), squeeze=False)
    for p in range(N_PLANTS):
        for j in range(nq):
            ax = axes[p, j]
            ax.plot(t_x, refs[p][:, j], "k--", lw=1.5, label="reference")
            ax.axhline(refs[p][-1, j], color="0.6", ls=":", alpha=0.7)
            for name in names:
                ax.plot(t_x, rollouts[(name, p)][:, j], color=COLORS.get(name, "tab:purple"),
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


def plot_cartesian(cfg, names, refs, rollouts, save_path):
    """Grid [plant x controller] of strobe stick-figures; red = target pose."""
    nq = cfg.NQ
    bound = LINK_LENGTH * cfg.N_LINKS + 0.3
    fig, axes = plt.subplots(N_PLANTS, len(names),
                             figsize=(4 * len(names), 4 * N_PLANTS), squeeze=False)
    for p in range(N_PLANTS):
        T = refs[p].shape[0] - 1
        snaps = np.linspace(0, T, N_STROBE).astype(int)
        target_xs, target_zs = forward_kinematics(refs[p][-1, :nq][None])    # leading dim for FK
        for c, name in enumerate(names):
            ax = axes[p, c]
            q = rollouts[(name, p)][:, :nq]                  # (T+1, nq)
            xs, zs = forward_kinematics(q)                    # (T+1, N+1)
            ax.plot(xs[:, -1], zs[:, -1], color=COLORS.get(name, "tab:purple"),
                    lw=1, alpha=0.5, label="tip path")
            for k, s in enumerate(snaps):
                a = 0.12 + 0.88 * (k / (len(snaps) - 1))
                ax.plot(xs[s], zs[s], "-o", color="0.2", alpha=a, ms=3, lw=1.5)
            ax.plot(target_xs[0], target_zs[0], "-o", color="tab:red", lw=2, ms=5, label="target")
            ax.plot(0, 0, "ks", ms=6)
            ax.set_aspect("equal")
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
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
    cfg = training.load_config()
    traj_path = cfg.OUTPUT_DIR / "trajectories.npz"
    timeseries_path = cfg.OUTPUT_DIR / "eval_rollouts_timeseries.png"
    cartesian_path = cfg.OUTPUT_DIR / "eval_rollouts_cartesian.png"

    x_refs, u_refs = load_trajectories(traj_path)
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    w = cfg.PURE["n_history"]

    print("scanning for available controller params ...")
    factories = build_factories(cfg, cfg.OUTPUT_DIR)
    names = list(factories.keys())
    if len(names) <= 1:
        print("only PD available — train at least one controller first")
        return

    mjx_model_nominal, nominal_body_mass = build_mjx_model(cfg.MODEL_PATH)
    rollout_fns = {name: make_rollout_fn(cfg, mjx_model_nominal, nominal_body_mass, fac, T, w)
                   for name, fac in factories.items()}

    # Sample plants from the eval seed family.
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), N_PLANTS)
    thetas = [sample_theta(k, cfg.N_LINKS, cfg.DR_RANGES) for k in theta_keys]
    idxs = [p % n_traj for p in range(N_PLANTS)]

    refs = [np.asarray(x_refs[idxs[p]]) for p in range(N_PLANTS)]
    rollouts = {}
    print(f"rolling out {N_PLANTS} plants x {len(names)} controllers (T={T}) ...")
    for p in range(N_PLANTS):
        idx = idxs[p]
        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]
        x_target = x_refs[idx, -1]
        msg = f"  plant {p} (traj {idx}, theta={np.round(np.asarray(thetas[p]), 2)})"
        for name in names:
            xs_full, vs = rollout_fns[name](thetas[p], x_ref_full, u_ref_full)
            ep = float(losses.endpoint_error(xs_full[-1], x_target, nq))
            vrms = float(jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1))))
            rollouts[(name, p)] = np.asarray(xs_full)
            msg += f"\n      {name:10s} endpoint={ep:.2f}  |v|rms={vrms:.2f}"
        print(msg)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeseries(cfg, names, refs, rollouts, timeseries_path)
    plot_cartesian(cfg, names, refs, rollouts, cartesian_path)


if __name__ == "__main__":
    main()
