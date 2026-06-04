"""Sanity-check supervised labels: open-loop apply u_nom vs u_nom + u_residual_label, plot tracking."""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

from lib.domain_randomization import apply_theta


N_TRAJ_SHOW = 3
N_THETA_SHOW = 2


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("plant_cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def open_loop_factory(mjx_model):
    nq = mjx_model.nq

    def open_loop(x_init, u_seq):
        d0 = mjx.make_data(mjx_model)
        d0 = d0.replace(qpos=x_init[:nq], qvel=x_init[nq:])
        d0 = mjx.forward(mjx_model, d0)

        def step(d, u):
            d = d.replace(ctrl=u)
            d = mjx.step(mjx_model, d)
            return d, jnp.concatenate([d.qpos, d.qvel])

        _, xs = jax.lax.scan(step, d0, u_seq)
        return xs

    return jax.jit(open_loop)


def closed_loop_factory(mjx_model, kp, kd):
    """Apply feedforward u_seq[t] plus PD feedback kp*(x_ref-x)+kd*(...) each step."""
    nq = mjx_model.nq

    def closed_loop(x_init, u_seq, x_ref_seq):
        d0 = mjx.make_data(mjx_model)
        d0 = d0.replace(qpos=x_init[:nq], qvel=x_init[nq:])
        d0 = mjx.forward(mjx_model, d0)

        def step(d, inp):
            u_ff, x_ref = inp
            pd = kp * (x_ref[:nq] - d.qpos) + kd * (x_ref[nq:] - d.qvel)
            d = d.replace(ctrl=u_ff + pd)
            d = mjx.step(mjx_model, d)
            return d, jnp.concatenate([d.qpos, d.qvel])

        _, xs = jax.lax.scan(step, d0, (u_seq, x_ref_seq))
        return xs

    return jax.jit(closed_loop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    nq = cfg.NQ
    n_links = cfg.N_LINKS
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    dataset = np.load(cfg.OUTPUT_DIR / "supervised_dataset.npz")
    x_refs = dataset["x_refs"]
    u_refs = dataset["u_refs"]
    thetas = dataset["thetas"]
    u_residual_labels = dataset["u_residual_labels"]

    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model_nominal = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model_nominal.body_mass)

    fig, axes = plt.subplots(
        N_TRAJ_SHOW * N_THETA_SHOW, nq,
        figsize=(4 * nq, 2.5 * N_TRAJ_SHOW * N_THETA_SHOW),
        squeeze=False,
    )

    for ti in range(N_TRAJ_SHOW):
        traj_idx = ti
        x_init = x_refs[traj_idx, 0]
        x_nom = x_refs[traj_idx]
        u_nom = u_refs[traj_idx]
        T = u_nom.shape[0]
        t = np.arange(T + 1) * cfg.TIMESTEP

        x_ref_seq = jnp.asarray(x_nom[:T])

        for thi in range(N_THETA_SHOW):
            theta_idx = thi
            theta = jnp.asarray(thetas[theta_idx])
            mjx_perturbed = apply_theta(mjx_model_nominal, theta, nominal_body_mass, n_links)
            open_loop = open_loop_factory(mjx_perturbed)
            closed_loop = closed_loop_factory(mjx_perturbed, kp, kd)

            u_residual = u_residual_labels[traj_idx, theta_idx]
            u_ff = jnp.asarray(u_nom)
            u_ff_resid = jnp.asarray(u_nom + u_residual)
            x0 = jnp.asarray(x_init)

            xs_unom = np.asarray(open_loop(x0, u_ff))
            xs_resid = np.asarray(open_loop(x0, u_ff_resid))
            xs_pd = np.asarray(closed_loop(x0, u_ff, x_ref_seq))
            xs_resid_pd = np.asarray(closed_loop(x0, u_ff_resid, x_ref_seq))

            xs_unom = np.vstack([x_init[None], xs_unom])
            xs_resid = np.vstack([x_init[None], xs_resid])
            xs_pd = np.vstack([x_init[None], xs_pd])
            xs_resid_pd = np.vstack([x_init[None], xs_resid_pd])

            row = ti * N_THETA_SHOW + thi
            for j in range(nq):
                ax = axes[row, j]
                ax.plot(t, x_nom[:, j], "k", lw=1.5, label="x_nom")
                ax.plot(t, xs_unom[:, j], "tab:red", lw=1, label="u_nom only")
                ax.plot(t, xs_resid[:, j], "tab:green", lw=1, label="u_nom + resid")
                ax.plot(t, xs_pd[:, j], "tab:blue", lw=1, label="u_nom + pd")
                ax.plot(t, xs_resid_pd[:, j], "tab:orange", lw=1, label="u_nom + resid + pd")
                ax.set_title(f"traj {traj_idx}, theta {theta_idx}, q{j+1}")
                ax.set_xlabel("t (s)")
                ax.set_ylabel("rad")
                ax.grid(True, alpha=0.3)
                if row == 0 and j == 0:
                    ax.legend(fontsize=8)

    fig.tight_layout()
    out = cfg.OUTPUT_DIR / "residual_check.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    n_cases = N_TRAJ_SHOW * N_THETA_SHOW
    err_unom = np.zeros(n_cases)
    err_resid = np.zeros(n_cases)
    err_pd = np.zeros(n_cases)
    err_resid_pd = np.zeros(n_cases)
    for ti in range(N_TRAJ_SHOW):
        x_init = x_refs[ti, 0]
        x_nom = x_refs[ti]
        u_nom = u_refs[ti]
        T = u_nom.shape[0]
        x_ref_seq = jnp.asarray(x_nom[:T])
        x0 = jnp.asarray(x_init)
        for thi in range(N_THETA_SHOW):
            theta = jnp.asarray(thetas[thi])
            mjx_perturbed = apply_theta(mjx_model_nominal, theta, nominal_body_mass, n_links)
            open_loop = open_loop_factory(mjx_perturbed)
            closed_loop = closed_loop_factory(mjx_perturbed, kp, kd)
            u_ff = jnp.asarray(u_nom)
            u_ff_resid = jnp.asarray(u_nom + u_residual_labels[ti, thi])

            xs_unom = np.asarray(open_loop(x0, u_ff))
            xs_resid = np.asarray(open_loop(x0, u_ff_resid))
            xs_pd = np.asarray(closed_loop(x0, u_ff, x_ref_seq))
            xs_resid_pd = np.asarray(closed_loop(x0, u_ff_resid, x_ref_seq))

            idx = ti * N_THETA_SHOW + thi
            err_unom[idx] = np.linalg.norm(xs_unom[-1, :nq] - x_nom[-1, :nq])
            err_resid[idx] = np.linalg.norm(xs_resid[-1, :nq] - x_nom[-1, :nq])
            err_pd[idx] = np.linalg.norm(xs_pd[-1, :nq] - x_nom[-1, :nq])
            err_resid_pd[idx] = np.linalg.norm(xs_resid_pd[-1, :nq] - x_nom[-1, :nq])

    print(f"final qpos error vs x_nom:")
    print(f"  u_nom only:         mean={err_unom.mean():.4f}  max={err_unom.max():.4f}")
    print(f"  u_nom + resid:      mean={err_resid.mean():.4f}  max={err_resid.max():.4f}")
    print(f"  u_nom + pd:         mean={err_pd.mean():.4f}  max={err_pd.max():.4f}")
    print(f"  u_nom + resid + pd: mean={err_resid_pd.mean():.4f}  max={err_resid_pd.max():.4f}")


if __name__ == "__main__":
    main()
