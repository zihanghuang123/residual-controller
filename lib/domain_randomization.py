"""Plant parameter sampling: defines the DR distribution p(theta) and samples theta vectors.

Theta layout (flat vector, length 3 * n_links):
    [0 .. n_links)              -- mass_scale per link 
    [n_links .. 2*n_links)      -- damping per link 
    [2*n_links .. 3*n_links)    -- frictionloss per link 
"""

import jax
import jax.numpy as jnp


def sample_theta(key: jax.Array, n_links: int, dr_ranges: dict) -> jnp.ndarray:
    """Sample one theta vector uniformly from the DR distribution."""
    k_mass, k_damp, k_fric = jax.random.split(key, 3)
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
    return jnp.concatenate([mass, damping, frictionloss])


def apply_theta(mjx_model, theta: jnp.ndarray, nominal_body_mass: jnp.ndarray, n_links: int):
    """Return a new mjx.Model with theta applied."""
    mass_scale = theta[:n_links]
    damping = theta[n_links:2 * n_links]
    frictionloss = theta[2 * n_links:3 * n_links]

    # Scale body masses (skip body 0 = world).
    new_body_mass = nominal_body_mass.at[1:1 + n_links].set(nominal_body_mass[1:1 + n_links] * mass_scale)

    return mjx_model.replace(
        body_mass=new_body_mass,
        dof_damping=damping,
        dof_frictionloss=frictionloss,
    )
