"""Predict the trainable BPTT horizon from the closed-loop Lyapunov exponent.

Largest Lyapunov exponent of the PD-closed-loop dynamics along the reference (jvp power
iteration through the same mjx.step BPTT uses), averaged over DR plants. The Lyapunov time
tau = 1/lambda (in steps) sets the gradient-variance scale: variance ~ exp(2*lambda*H), so
the usable horizon is O(tau) and gradients degrade past a few tau.
"""

import sys
from functools import partial
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

MEASURE_STEPS = 1500   # steps to average the exponent over (capped at T)
N_SAMPLES = 100          # (trajectory, theta) draws to average lambda over


def step_fn(model, x, x_ref_t, u_ref_t, kp, kd, nq):
    """One PD-closed-loop MJX step (residual v=0): x -> x_next, matching rollout's control law."""
    qpos, qvel = x[:nq], x[nq:]
    d = rollout.make_initial_data(model, qpos, qvel)
    pd = kp * (x_ref_t[:nq] - qpos) + kd * (x_ref_t[nq:] - qvel)
    d = d.replace(ctrl=u_ref_t + pd)
    d = mjx.step(model, d)
    return jnp.concatenate([d.qpos, d.qvel])


def lyapunov(model, x_ref, u_ref, v0, kp, kd, nq, n_steps):
    """Per-step largest Lyapunov exponent: power-iterate the tangent via jvp, accumulate log growth."""
    def body(carry, t):
        x, v = carry
        f = lambda z: step_fn(model, z, x_ref[t], u_ref[t], kp, kd, nq)
        x_next, v_next = jax.jvp(f, (x,), (v,))
        norm = jnp.linalg.norm(v_next)
        return (x_next, v_next / norm), jnp.log(norm)

    (_, _), logs = jax.lax.scan(body, (x_ref[0], v0), jnp.arange(n_steps))
    return jnp.sum(logs) / n_steps


def main():
    cfg = training.load_config()
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    n_steps = min(MEASURE_STEPS, T)
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    lyap = jax.jit(partial(lyapunov, kp=kp, kd=kd, nq=nq, n_steps=n_steps))

    key = jax.random.PRNGKey(cfg.EVAL_SEED)
    print(f"estimating lambda over {N_SAMPLES} plants, {n_steps} steps each ...")
    lams = []
    for s in range(N_SAMPLES):
        key, ik, tk, vk = jax.random.split(key, 4)
        idx = int(jax.random.randint(ik, (), 0, n_traj))
        theta = sample_theta(tk, cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        v0 = jax.random.normal(vk, (2 * nq,))
        v0 = v0 / jnp.linalg.norm(v0)
        lam = float(lyap(model, x_refs[idx], u_refs[idx], v0))
        lams.append(lam)
        print(f"  sample {s}: traj {idx}  lambda = {lam:+.5f} /step")

    lams = np.asarray(lams)
    finite = lams[np.isfinite(lams)]
    n_div = int((~np.isfinite(lams)).sum())
    lam_mean = float(finite.mean()) if finite.size else float("nan")
    lam_med = float(np.median(finite)) if finite.size else float("nan")

    def _tau(lam):
        return f"tau = {1.0 / lam:.0f} steps ({1.0 / lam * cfg.TIMESTEP:.3f} s)" if lam > 0 else "contracting (lambda <= 0)"

    print(f"\nover {finite.size} finite samples ({n_div} diverged):")
    print(f"  mean   lambda = {lam_mean:+.5f} /step  ->  {_tau(lam_mean)}")
    print(f"  median lambda = {lam_med:+.5f} /step  ->  {_tau(lam_med)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(finite, bins=30, color="tab:blue", alpha=0.8)
    ax.axvline(lam_mean, color="tab:red", ls="--", lw=1.5, label=f"mean {lam_mean:.4f}")
    ax.axvline(lam_med, color="tab:green", ls="--", lw=1.5, label=f"median {lam_med:.4f}")
    ax.set_xlabel("per-plant lambda (/step)")
    ax.set_ylabel("count")
    ax.set_title(f"{cfg.PLANT_NAME}: per-plant lambda ({finite.size} plants, {n_div} diverged)")
    ax.legend()
    fig.tight_layout()
    out = cfg.OUTPUT_DIR / "lambda_hist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
