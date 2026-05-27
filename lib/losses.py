"""Loss functions: tracking, control regularization, theta estimation, endpoint metric."""

import jax.numpy as jnp


def wrap_angle(diff: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle difference to (-pi, pi]."""
    return jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi


def tracking_loss(x: jnp.ndarray, x_ref: jnp.ndarray, nq: int) -> jnp.ndarray:
    """Mean squared tracking error per timestep; angle differences wrapped to (-pi, pi]."""
    q_diff = wrap_angle(x[..., :nq] - x_ref[..., :nq])
    qvel_diff = x[..., nq:] - x_ref[..., nq:]
    return jnp.mean(jnp.sum(q_diff ** 2, axis=-1) + jnp.sum(qvel_diff ** 2, axis=-1))


def reg_loss(v: jnp.ndarray) -> jnp.ndarray:
    """Mean squared residual control norm over the trajectory."""
    return jnp.mean(jnp.sum(v ** 2, axis=-1))


def theta_loss(theta_pred: jnp.ndarray, theta_true: jnp.ndarray) -> jnp.ndarray:
    """MSE between predicted and true theta vectors."""
    return jnp.mean((theta_pred - theta_true) ** 2)


def endpoint_error(x: jnp.ndarray, x_ref: jnp.ndarray, nq: int) -> jnp.ndarray:
    """L2 distance between final state and reference state; angles wrapped."""
    q_diff = wrap_angle(x[:nq] - x_ref[:nq])
    qvel_diff = x[nq:] - x_ref[nq:]
    return jnp.sqrt(jnp.sum(q_diff ** 2) + jnp.sum(qvel_diff ** 2))
