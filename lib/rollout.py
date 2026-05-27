"""MJX rollout helpers.

One unified `rollout` function steps physics under a closed-loop control law
    u(t) = u*(t) + PD(x*(t) - x(t)) + controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window)
while maintaining sliding history windows of length w for both simulated and reference data.

For PD-only rollouts, pass `lambda *_: jnp.zeros((nu,))`.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from mujoco import mjx


def make_initial_data(mjx_model, qpos: jnp.ndarray, qvel: jnp.ndarray):
    """Allocate an MjxData at the given state, with derived quantities populated."""
    d = mjx.make_data(mjx_model)
    d = d.replace(qpos=qpos, qvel=qvel)
    return mjx.forward(mjx_model, d)


def make_network_input(x_hist_full: jnp.ndarray,
                       u_hist: jnp.ndarray,
                       x_ref_window: Optional[jnp.ndarray] = None,
                       u_ref_window: Optional[jnp.ndarray] = None,
                       theta_estimate: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Flatten history (+ optional reference window, + optional theta) into 1D vector."""
    parts = [x_hist_full.flatten(), u_hist.flatten()]
    if x_ref_window is not None:
        parts.append(x_ref_window.flatten())
    if u_ref_window is not None:
        parts.append(u_ref_window.flatten())
    if theta_estimate is not None:
        parts.append(theta_estimate)
    return jnp.concatenate(parts)


def rollout(mjx_model,
            x_init: jnp.ndarray,
            x_refs: jnp.ndarray,
            u_refs: jnp.ndarray,
            x_hist0: jnp.ndarray,
            u_hist0: jnp.ndarray,
            x_ref_hist0: jnp.ndarray,
            u_ref_hist0: jnp.ndarray,
            controller_fn: Callable,
            kp: jnp.ndarray,
            kd: jnp.ndarray,
            n_steps: int):
    """Roll out closed-loop control under MJX."""
    nq = mjx_model.nq
    d0 = make_initial_data(mjx_model, x_init[:nq], x_init[nq:])

    def step(carry, t):
        d, x_hist, u_hist, x_ref_hist, u_ref_hist = carry
        x_curr = jnp.concatenate([d.qpos, d.qvel])

        x_ref = x_refs[t]
        u_ref = u_refs[t]

        x_hist_full = jnp.concatenate([x_hist, x_curr[None]], axis=0)
        x_ref_window = jnp.concatenate([x_ref_hist, x_ref[None]], axis=0)
        u_ref_window = jnp.concatenate([u_ref_hist, u_ref[None]], axis=0)

        v = controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window)
        pd = kp * (x_ref[:nq] - d.qpos) + kd * (x_ref[nq:] - d.qvel)
        u = u_ref + pd + v

        d = d.replace(ctrl=u)
        d = mjx.step(mjx_model, d)

        new_x_hist = jnp.concatenate([x_hist[1:], x_curr[None]], axis=0)
        new_u_hist = jnp.concatenate([u_hist[1:], u[None]], axis=0)
        new_x_ref_hist = jnp.concatenate([x_ref_hist[1:], x_ref[None]], axis=0)
        new_u_ref_hist = jnp.concatenate([u_ref_hist[1:], u_ref[None]], axis=0)

        return (d, new_x_hist, new_u_hist, new_x_ref_hist, new_u_ref_hist), (x_curr, u, v)

    init_carry = (d0, x_hist0, u_hist0, x_ref_hist0, u_ref_hist0)
    final_carry, (states, controls, residuals) = jax.lax.scan(step, init_carry, jnp.arange(n_steps))
    final_d = final_carry[0]
    x_final = jnp.concatenate([final_d.qpos, final_d.qvel])
    return states, controls, residuals, x_final
