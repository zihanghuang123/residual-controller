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


class _GRU(nn.Module):
    """Stacked GRU + linear output head; output head zero-init optional."""
    hidden_sizes: Sequence[int]
    out_dim: int
    final_zero_init: bool = False

    def setup(self):
        self.cells = [nn.GRUCell(features=h) for h in self.hidden_sizes]
        kernel_init = nn.initializers.zeros if self.final_zero_init else nn.initializers.lecun_normal()
        self.out_layer = nn.Dense(self.out_dim, kernel_init=kernel_init)

    def __call__(self, h, x: jnp.ndarray):
        new_h_list = []
        out = x
        for i, cell in enumerate(self.cells):
            new_h_i, out = cell(h[i], out)
            new_h_list.append(new_h_i)
        return tuple(new_h_list), self.out_layer(out)


class GRUPureController(nn.Module):
    """
    Input:  carried hidden state h, per-step feature vector x
            (caller assembles, e.g. concat(x_t, u_{t-1}, x_ref_t, u_ref_t))
    Output: new hidden state, control residual v in R^NU
    """
    hidden_sizes: Sequence[int]
    out_dim: int

    def setup(self):
        self.gru = _GRU(self.hidden_sizes, self.out_dim, final_zero_init=True)

    def __call__(self, h, x: jnp.ndarray):
        return self.gru(h, x)


class GRUController(nn.Module):
    """
    Per-step RNN residual controller with the theta estimate folded into x.
    Architecturally identical to GRUPureController.
    """
    hidden_sizes: Sequence[int]
    out_dim: int

    def setup(self):
        self.gru = _GRU(self.hidden_sizes, self.out_dim, final_zero_init=True)

    def __call__(self, h, x: jnp.ndarray):
        return self.gru(h, x)


class GRUThetaEstimator(nn.Module):
    """
    Input:  carried hidden state h, per-step feature vector x
            (caller assembles, e.g. concat(x_t, u_{t-1}))
    Output: new hidden state, theta_estimate in R^THETA_DIM
    """
    hidden_sizes: Sequence[int]
    theta_dim: int

    def setup(self):
        self.gru = _GRU(self.hidden_sizes, self.theta_dim, final_zero_init=False)

    def __call__(self, h, x: jnp.ndarray):
        return self.gru(h, x)


def gru_initial_state(hidden_sizes: Sequence[int]):
    """Zero h_0 for a stacked GRU. Per-layer shapes match hidden_sizes (supports mixed widths)."""
    return tuple(jnp.zeros((h,)) for h in hidden_sizes)
