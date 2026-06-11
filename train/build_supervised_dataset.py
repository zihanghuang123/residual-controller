"""Supervised residual labels: inverse(model_theta, q_pert, qd_pert, qddot_nom) - u_nom."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from lib import training
from lib.domain_randomization import apply_theta, sample_theta

N_THETAS = 50      # DR plants per trajectory
SEED = 42
NOISE_Q = 0.1      # qpos perturbation std
NOISE_QD = 0.5     # qvel perturbation std


def main():
    cfg = training.load_config()
    nq, nu, n_links = cfg.NQ, cfg.NU, cfg.N_LINKS

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

    key = jax.random.PRNGKey(SEED)
    key, theta_root = jax.random.split(key)
    theta_keys = jax.random.split(theta_root, N_THETAS)
    thetas = jnp.stack([sample_theta(k, n_links, cfg.DR_RANGES) for k in theta_keys])

    perturbed_models = jax.vmap(
        apply_theta, in_axes=(None, 0, None, None)
    )(mjx_model_nominal, thetas, nominal_body_mass, n_links)

    inverse_over_time = jax.vmap(training.inverse_dynamics, in_axes=(None, 0, 0, 0))
    inverse_over_theta_time = jax.jit(
        jax.vmap(inverse_over_time, in_axes=(0, None, None, None))
    )

    noise_keys = jax.random.split(key, N)
    x_perturbed = np.zeros((N, T, 2 * nq), dtype=np.float32)
    u_residual_labels = np.zeros((N, N_THETAS, T, nu), dtype=np.float32)

    print(f"computing labels for {N} trajectories x {N_THETAS} thetas x {T} steps")
    for i in range(N):
        q_nom = x_refs[i, :T, :nq]
        qd_nom = x_refs[i, :T, nq:]
        qddot = (x_refs[i, 1:T + 1, nq:] - qd_nom) / cfg.TIMESTEP
        u_nom = u_refs[i]

        kq, kqd = jax.random.split(noise_keys[i])
        q_pert = q_nom + NOISE_Q * jax.random.normal(kq, q_nom.shape)
        qd_pert = qd_nom + NOISE_QD * jax.random.normal(kqd, qd_nom.shape)

        tau_true = inverse_over_theta_time(perturbed_models, q_pert, qd_pert, qddot)

        x_perturbed[i] = np.asarray(jnp.concatenate([q_pert, qd_pert], axis=-1))
        u_residual_labels[i] = np.asarray(tau_true) - np.asarray(u_nom)[None, :, :]

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N}")

    out = cfg.OUTPUT_DIR / "supervised_dataset.npz"
    np.savez(
        out,
        x_refs=np.asarray(x_refs),
        u_refs=np.asarray(u_refs),
        x_perturbed=x_perturbed,
        thetas=np.asarray(thetas),
        u_residual_labels=u_residual_labels,
    )
    print(f"saved {out}  "
          f"({u_residual_labels.nbytes / 1e6:.1f} MB labels, "
          f"{x_perturbed.nbytes / 1e6:.1f} MB x_perturbed)")


if __name__ == "__main__":
    main()
