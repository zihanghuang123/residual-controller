"""Sanity-check solved TO trajectories — no GPU / MuJoCo / display needed.

Reads outputs/<plant>/trajectories.npz and writes two PNGs:
    trajectories_timeseries.png  joint angles (vs target) + control torques vs time
    trajectories_cartesian.png   strobe stick-figure of the swing-up in the x-z plane
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # headless WSL: render to file, never to a window
import matplotlib.pyplot as plt
import numpy as np

from double_pendulum import config as cfg

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
TRAJ_PATH = OUTPUT_DIR / "trajectories.npz"
TIMESERIES_FIG_PATH = OUTPUT_DIR / "trajectories_timeseries.png"
CARTESIAN_FIG_PATH = OUTPUT_DIR / "trajectories_cartesian.png"

L1 = L2 = 1.0   # link lengths from model.xml
N_SHOW = 4      # how many converged trajectories to plot
N_STROBE = 14   # stick-figure snapshots per cartesian panel


def forward_kinematics(q: np.ndarray):
    """Joint angles (..., 2) -> (x1, z1, x2, z2): elbow and tip in the x-z plane.

    At q=0 both links hang straight down (-z); +q rotates about +y. q1=pi is 'up'.
    """
    q1 = q[..., 0]
    q12 = q[..., 0] + q[..., 1]
    x1 = -L1 * np.sin(q1)
    z1 = -L1 * np.cos(q1)
    x2 = x1 - L2 * np.sin(q12)
    z2 = z1 - L2 * np.cos(q12)
    return x1, z1, x2, z2


def load():
    data = np.load(TRAJ_PATH)
    return (data["x_refs"], data["u_refs"], data["x_inits"],
            data["x_targets"], data["converged"].astype(bool))


def ctrl_limit():
    """The |ctrl| MJX will clamp to, read from the model (None if actuators unlimited).

    Crocoddyl ignores this limit when solving; MJX enforces it at rollout time. If the
    feedforward exceeds it, u* is silently clamped during training and x* is infeasible.
    """
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    except Exception as e:   # mujoco missing / model unreadable — skip the check, keep plots
        print(f"  (could not read ctrl limit from model: {e})")
        return None
    if not m.actuator_ctrllimited.any():
        return None
    return float(np.abs(m.actuator_ctrlrange).max())


def plot_timeseries(x_refs, u_refs, x_targets, idxs, save_path):
    """One row per trajectory: angles (with target lines) | control torques."""
    nq = cfg.NQ
    dt = cfg.TIMESTEP
    t_x = np.arange(x_refs.shape[1]) * dt
    t_u = np.arange(u_refs.shape[1]) * dt

    fig, axes = plt.subplots(len(idxs), 2, figsize=(12, 3 * len(idxs)), squeeze=False)
    for row, idx in enumerate(idxs):
        ax_q, ax_u = axes[row]
        for j in range(nq):
            line, = ax_q.plot(t_x, x_refs[idx, :, j], label=f"q{j + 1}")
            ax_q.axhline(x_targets[idx, j], color=line.get_color(),
                         ls="--", alpha=0.5, label=f"q{j + 1} target")
        ax_q.set_ylabel("angle (rad)")
        ax_q.set_title(f"traj {idx}: joint angles")
        ax_q.legend(fontsize=8, ncol=2)
        ax_q.grid(True, alpha=0.3)

        for j in range(cfg.NU):
            ax_u.plot(t_u, u_refs[idx, :, j], label=f"u{j + 1}")
        ax_u.set_ylabel("torque")
        ax_u.set_title(f"traj {idx}: control")
        ax_u.legend(fontsize=8)
        ax_u.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_cartesian(x_refs, x_targets, idxs, save_path):
    """Strobe stick-figures: faint=early, dark=late; tip trace; start/target marked."""
    fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4.5), squeeze=False)
    axes = axes[0]
    T = x_refs.shape[1] - 1
    snaps = np.linspace(0, T, N_STROBE).astype(int)

    for ax, idx in zip(axes, idxs):
        q = x_refs[idx, :, :cfg.NQ]
        x1, z1, x2, z2 = forward_kinematics(q)

        ax.plot(x2, z2, color="tab:blue", lw=1, alpha=0.6, label="tip path")
        for k, s in enumerate(snaps):
            a = 0.12 + 0.88 * (k / (len(snaps) - 1))
            ax.plot([0, x1[s], x2[s]], [0, z1[s], z2[s]],
                    "-o", color="0.2", alpha=a, ms=3, lw=1.5)

        # target configuration
        tx1, tz1, tx2, tz2 = forward_kinematics(x_targets[idx, :cfg.NQ])
        ax.plot([0, tx1, tx2], [0, tz1, tz2], "-o", color="tab:red",
                lw=2, ms=5, label="target")
        ax.plot(0, 0, "ks", ms=6)  # fixed base

        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_title(f"traj {idx}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("swing-up (faint = start, dark = end)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    x_refs, u_refs, _x_inits, x_targets, converged = load()
    nq = cfg.NQ
    conv_idxs = np.where(converged)[0]
    print(f"{len(conv_idxs)}/{len(converged)} trajectories converged")

    # Final angle error (wrapped) vs target, across all converged trajectories.
    final_q = x_refs[conv_idxs, -1, :nq]
    target_q = x_targets[conv_idxs, :nq]
    err = np.mod(final_q - target_q + np.pi, 2 * np.pi) - np.pi
    rms = np.sqrt((err ** 2).sum(axis=1))
    print(f"final angle error vs target (rad): "
          f"mean={rms.mean():.4f}  med={np.median(rms):.4f}  max={rms.max():.4f}")

    # Feedforward feasibility: does u* stay within what MJX will actually apply?
    peak_per_traj = np.abs(u_refs[conv_idxs]).max(axis=(1, 2))
    limit = ctrl_limit()
    print(f"max |u_ref| over converged trajectories: {peak_per_traj.max():.1f}")
    if limit is None:
        print("  actuators unlimited (ctrllimited off) — MJX will not clamp the feedforward")
    else:
        n_over = int((peak_per_traj > limit).sum())
        status = "OK" if n_over == 0 else f"*** {n_over} trajectories EXCEED it — raise ctrlrange ***"
        print(f"  actuator limit ±{limit:.0f}  ->  {status}")

    idxs = conv_idxs[:N_SHOW]
    print(f"plotting trajectories {list(idxs)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeseries(x_refs, u_refs, x_targets, idxs, TIMESERIES_FIG_PATH)
    plot_cartesian(x_refs, x_targets, idxs, CARTESIAN_FIG_PATH)


if __name__ == "__main__":
    main()
