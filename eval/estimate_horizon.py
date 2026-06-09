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
from mujoco import mjx

from lib import rollout, training
from lib.domain_randomization import apply_theta, sample_theta

MEASURE_STEPS = 1500   # steps to average the exponent over (capped at T)
N_SAMPLES = 20          # (trajectory, theta) draws to average lambda over
SWEET_LO, SWEET_HI = 0.5, 1.5   # predicted sweet-spot band, in units of tau
DEGRADE = 2.0                   # expect gradient degradation beyond this many tau


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

    lam_mean = sum(lams) / len(lams)
    print(f"\nmean lambda = {lam_mean:+.5f} /step")
    if lam_mean <= 0:
        print("closed loop is (on average) contracting: no positive exponent. Gradient blow-up, if any,\n"
              "is from transient/non-normal growth, not asymptotic chaos -- this predictor doesn't apply.")
        return

    tau = 1.0 / lam_mean
    print(f"Lyapunov time tau = 1/lambda = {tau:.0f} steps  ({tau * cfg.TIMESTEP:.3f} s)")
    print(f"predicted sweet spot ~ [{SWEET_LO * tau:.0f}, {SWEET_HI * tau:.0f}] steps; "
          f"expect degradation beyond ~{DEGRADE * tau:.0f} steps")
    print("\ngradient-variance amplification exp(2*lambda*H) at candidate horizons:")
    for H in (250, 500, 1000, 2000):
        print(f"  H={H:5d}:  x{jnp.exp(2 * lam_mean * H):.2e}")


if __name__ == "__main__":
    main()
