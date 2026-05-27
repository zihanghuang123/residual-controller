"""Neural network architectures: controllers and theta estimators."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class _MLP(nn.Module):
    """Generic MLP: tanh hidden layers, linear output. Final layer optionally zero-init."""
    hidden_sizes: Sequence[int]
    out_dim: int
    final_zero_init: bool = False

    def setup(self):
        self.hidden_layers = [nn.Dense(h) for h in self.hidden_sizes]
        kernel_init = nn.initializers.zeros if self.final_zero_init else nn.initializers.lecun_normal()
        self.out_layer = nn.Dense(self.out_dim, kernel_init=kernel_init)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = nn.tanh(layer(x))
        return self.out_layer(x)


class MLPPureController(nn.Module):
    """
    Input:  history (x, u) + reference (x*, u*)
    Output: control residual v in R^NU
    """
    hidden_sizes: Sequence[int]
    out_dim: int

    def setup(self):
        self.mlp = _MLP(self.hidden_sizes, self.out_dim, final_zero_init=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)


class MLPController(nn.Module):
    """
    Input:  history (x, u) + reference (x*, u*) + theta_estimate
    Output: control residual v in R^NU
    """
    hidden_sizes: Sequence[int]
    out_dim: int

    def setup(self):
        self.mlp = _MLP(self.hidden_sizes, self.out_dim, final_zero_init=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)


class MLPThetaEstimator(nn.Module):
    """
    Input:  history (x, u) + reference (x*, u*)
    Output: theta_estimate in R^THETA_DIM
    """
    hidden_sizes: Sequence[int]
    theta_dim: int

    def setup(self):
        self.mlp = _MLP(self.hidden_sizes, self.theta_dim, final_zero_init=False)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)
