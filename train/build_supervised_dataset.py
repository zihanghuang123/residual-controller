"""Compute supervised residual-control labels via inverse dynamics on DR-sampled plants."""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from lib.domain_randomization import apply_theta, sample_theta


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("plant_cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-thetas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    nq = cfg.NQ
    nu = cfg.NU
    n_links = cfg.N_LINKS

    mj_model = mujoco.MjModel.from_xml_path(str(cfg.MODEL_PATH))
    mjx_model_nominal = mjx.put_model(mj_model)
    nominal_body_mass = jnp.asarray(mjx_model_nominal.body_mass)

    data = np.load(cfg.OUTPUT_DIR / "trajectories.npz")
    converged = data["converged"].astype(bool)
    keep = np.where(converged)[0]
    x_refs = jnp.asarray(data["x_refs"][keep])
    u_refs = jnp.asarray(data["u_refs"][keep])
    print(f"using {len(keep)}/{len(converged)} converged trajectories")

    N, T_plus_1, _ = x_refs.shape
    T = u_refs.shape[1]
    assert T_plus_1 == T + 1, f"x_refs has {T_plus_1} frames, u_refs has {T} controls"

    key = jax.random.PRNGKey(args.seed)
    theta_keys = jax.random.split(key, args.n_thetas)
    thetas = jnp.stack([sample_theta(k, n_links, cfg.DR_RANGES) for k in theta_keys])

    perturbed_models = jax.vmap(
        apply_theta, in_axes=(None, 0, None, None)
    )(mjx_model_nominal, thetas, nominal_body_mass, n_links)

    def inverse_step(model, qp, qv, qa):
        d = mjx.make_data(model)
        d = d.replace(qpos=qp, qvel=qv, qacc=qa)
        d = mjx.inverse(model, d)
        return d.qfrc_inverse

    inverse_over_time = jax.vmap(inverse_step, in_axes=(None, 0, 0, 0))
    inverse_over_theta_time = jax.jit(
        jax.vmap(inverse_over_time, in_axes=(0, None, None, None))
    )

    qddot_noms = np.zeros((N, T, nq), dtype=np.float32)
    u_residual_labels = np.zeros((N, args.n_thetas, T, nu), dtype=np.float32)

    print(f"computing labels for {N} trajectories x {args.n_thetas} thetas x {T} steps")
    for i in range(N):
        q_nom = x_refs[i, :T, :nq]
        qd_nom = x_refs[i, :T, nq:]
        qd_nom_next = x_refs[i, 1:T + 1, nq:]
        qddot = (qd_nom_next - qd_nom) / cfg.TIMESTEP
        u_nom = u_refs[i]

        tau_true = inverse_over_theta_time(perturbed_models, q_nom, qd_nom, qddot)

        qddot_noms[i] = np.asarray(qddot)
        u_residual_labels[i] = np.asarray(tau_true) - np.asarray(u_nom)[None, :, :]

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N}")

    out = cfg.OUTPUT_DIR / "supervised_dataset.npz"
    np.savez(
        out,
        x_refs=np.asarray(x_refs),
        u_refs=np.asarray(u_refs),
        qddot_noms=qddot_noms,
        thetas=np.asarray(thetas),
        u_residual_labels=u_residual_labels,
    )
    print(f"saved {out}  "
          f"({u_residual_labels.nbytes / 1e6:.1f} MB labels, "
          f"{qddot_noms.nbytes / 1e6:.1f} MB qddot)")


if __name__ == "__main__":
    main()
