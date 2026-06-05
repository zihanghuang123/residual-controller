"""Sanity-check tracking on DR-sampled plants: open-loop u_nom vs u_nom+pd vs computed-torque expert."""

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
from mujoco import mjx

from lib import rollout, training
from lib.domain_randomization import apply_theta, sample_theta

N_TRAJ_SHOW = 3
N_THETA_SHOW = 2
SEED = 0


def main():
    cfg = training.load_config()
    nq = cfg.NQ
    dt = cfg.TIMESTEP
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    theta_keys = jax.random.split(jax.random.PRNGKey(SEED), N_THETA_SHOW)
    thetas = [sample_theta(k, cfg.N_LINKS, cfg.DR_RANGES) for k in theta_keys]

    def rollout_ctrl(model, x_init, ctrl_fn, n_steps):
        d0 = rollout.make_initial_data(model, x_init[:nq], x_init[nq:])

        def step(d, t):
            d = mjx.step(model, d.replace(ctrl=ctrl_fn(t, d.qpos, d.qvel)))
            return d, jnp.concatenate([d.qpos, d.qvel])

        _, xs = jax.lax.scan(step, d0, jnp.arange(n_steps))
        return np.asarray(xs)

    def expert_torque(model, q, qd, qddot):
        d = mjx.make_data(model).replace(qpos=q, qvel=qd, qacc=qddot)
        return mjx.inverse(model, d).qfrc_inverse

    fig, axes = plt.subplots(
        N_TRAJ_SHOW * N_THETA_SHOW, nq,
        figsize=(4 * nq, 2.5 * N_TRAJ_SHOW * N_THETA_SHOW), squeeze=False)

    print("final qpos error vs reference:")
    for ti in range(N_TRAJ_SHOW):
        x_nom = jnp.asarray(x_refs[ti])
        u_nom = jnp.asarray(u_refs[ti])
        T = u_nom.shape[0]
        x_init = x_nom[0]
        x_ref_t = x_nom[:T]
        qddot_ref = (x_nom[1:T + 1, nq:] - x_nom[:T, nq:]) / dt
        t_axis = np.arange(T + 1) * dt
        x_nom_np = np.asarray(x_nom)

        for thi in range(N_THETA_SHOW):
            model = apply_theta(mjx_model_nominal, thetas[thi], nominal_body_mass, cfg.N_LINKS)

            ol = rollout_ctrl(model, x_init, lambda t, q, qd: u_nom[t], T)
            pdc = rollout_ctrl(model, x_init,
                               lambda t, q, qd: u_nom[t] + kp * (x_ref_t[t, :nq] - q) + kd * (x_ref_t[t, nq:] - qd), T)
            exp = rollout_ctrl(model, x_init, lambda t, q, qd: expert_torque(model, q, qd, qddot_ref[t]), T)

            ol = np.vstack([np.asarray(x_init)[None], ol])
            pdc = np.vstack([np.asarray(x_init)[None], pdc])
            exp = np.vstack([np.asarray(x_init)[None], exp])

            row = ti * N_THETA_SHOW + thi
            for j in range(nq):
                ax = axes[row, j]
                ax.plot(t_axis, x_nom_np[:, j], "k", lw=1.5, label="reference")
                ax.plot(t_axis, ol[:, j], "tab:red", lw=1, label="u_nom (open loop)")
                ax.plot(t_axis, pdc[:, j], "tab:blue", lw=1, label="u_nom + pd")
                ax.plot(t_axis, exp[:, j], "tab:green", lw=1, label="expert (computed torque)")
                ax.set_title(f"traj {ti}, theta {thi}, q{j + 1}")
                ax.grid(True, alpha=0.3)
                if row == 0 and j == 0:
                    ax.legend(fontsize=8)

            e_ol = np.linalg.norm(ol[-1, :nq] - x_nom_np[-1, :nq])
            e_pd = np.linalg.norm(pdc[-1, :nq] - x_nom_np[-1, :nq])
            e_exp = np.linalg.norm(exp[-1, :nq] - x_nom_np[-1, :nq])
            print(f"  traj {ti} theta {thi}:  u_nom={e_ol:.3f}  pd={e_pd:.3f}  expert={e_exp:.4f}")

    fig.tight_layout()
    out = cfg.OUTPUT_DIR / "residual_check.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
