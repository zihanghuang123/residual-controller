"""Plant parameter sampling: defines the DR distribution p(theta) and samples theta vectors.

Theta layout (flat vector, length 4 * n_links):
    [0 .. n_links)              -- mass_scale per link
    [n_links .. 2*n_links)      -- damping offset per link
    [2*n_links .. 3*n_links)    -- frictionloss offset per link
    [3*n_links .. 4*n_links)    -- inertia_scale per link
"""

import jax
import jax.numpy as jnp


def sample_theta(key: jax.Array, n_links: int, dr_ranges: dict) -> jnp.ndarray:
    """Sample one theta vector uniformly from the DR distribution."""
    k_mass, k_damp, k_fric, k_inertia = jax.random.split(key, 4)
    mass = jax.random.uniform(
        k_mass, (n_links,),
        minval=dr_ranges["mass_scale"][0],
        maxval=dr_ranges["mass_scale"][1],
    )
    damping = jax.random.uniform(
        k_damp, (n_links,),
        minval=dr_ranges["damping"][0],
        maxval=dr_ranges["damping"][1],
    )
    frictionloss = jax.random.uniform(
        k_fric, (n_links,),
        minval=dr_ranges["frictionloss"][0],
        maxval=dr_ranges["frictionloss"][1],
    )
    inertia = jax.random.uniform(
        k_inertia, (n_links,),
        minval=dr_ranges["inertia_scale"][0],
        maxval=dr_ranges["inertia_scale"][1],
    )
    return jnp.concatenate([mass, damping, frictionloss, inertia])


def apply_theta(mjx_model, theta: jnp.ndarray, nominal_body_mass: jnp.ndarray, n_links: int):
    """Return a new mjx.Model with theta applied."""
    mass_scale = theta[:n_links]
    damping_offset = theta[n_links:2 * n_links]
    frictionloss_offset = theta[2 * n_links:3 * n_links]
    inertia_scale = theta[3 * n_links:4 * n_links]

    # Links are the last n_links bodies (skip world + any fixed base).
    b0 = mjx_model.nbody - n_links
    new_body_mass = nominal_body_mass.at[b0:b0 + n_links].set(nominal_body_mass[b0:b0 + n_links] * mass_scale)
    new_body_inertia = mjx_model.body_inertia.at[b0:b0 + n_links].set(
        mjx_model.body_inertia[b0:b0 + n_links] * inertia_scale[:, None])

    return mjx_model.replace(
        body_mass=new_body_mass,
        body_inertia=new_body_inertia,
        dof_damping=mjx_model.dof_damping + damping_offset,
        dof_frictionloss=mjx_model.dof_frictionloss + frictionloss_offset,
    )
