"""Pin the trainable BPTT horizon from the gradient noise scale (constant-free).

For each horizon H, measures the BPTT gradient noise scale  B_noise = tr(Cov)/||mean_grad||^2
over a batch of (theta, idx, t0) draws at fixed params. The batch averages the noise while
B_noise(H) <= batch_size; past that, longer H just adds noise the batch can't suppress and
training oscillates. So the trainable horizon is where B_noise(H) crosses cfg batch size.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from jax import tree_util

from lib import losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import GRUPureController, gru_initial_state

HORIZONS = (100, 250, 500, 750, 1000, 1500)
BATCH = 128 


def make_sample_grad(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs, H):
    """grad_params of the single-sample BPTT tracking loss at horizon H (matches train_pure_rnn)."""
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    hidden = cfg.PURE_RNN["hidden_sizes"]
    alpha = cfg.PURE_RNN["alpha_reg"]

    def loss(params, theta_key, idx, t0):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref_window = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H, 2 * nq))
        u_ref_window = jax.lax.dynamic_slice(u_refs[idx], (t0, 0), (H, cfg.NU))
        x_ref_for_loss = jax.lax.dynamic_slice(x_refs[idx], (t0, 0), (H + 1, 2 * nq))

        def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
            return network.apply(params, h, rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref))

        xs, _us, vs, x_final = rollout.rollout_rnn(
            model, x_ref_window[0], x_ref_window, u_ref_window,
            gru_initial_state(hidden), controller_fn, kp, kd, H)
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)
        return losses.tracking_loss(xs_full, x_ref_for_loss, nq) + alpha * losses.reg_loss(vs)

    return jax.grad(loss)


def noise_scale(grads):
    """Unbiased gradient noise scale tr(Cov)/||mean||^2; ||mean_hat||^2 is bias-corrected by
    -tr(Cov)/n (else it overestimates ||mean||^2 by tr(Cov)/n, capping the ratio at n)."""
    leaves = tree_util.tree_leaves(grads)
    n = leaves[0].shape[0]
    tr_cov = sum(g.var(axis=0, ddof=1).sum() for g in leaves)
    mean_sq = sum((g.mean(axis=0) ** 2).sum() for g in leaves) - tr_cov / n
    return float(tr_cov / mean_sq)


def main():
    cfg = training.load_config()
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    if params_path.exists():
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        print(f"using trained params {params_path}")
    else:
        nx = 2 * cfg.NQ
        params = network.init(jax.random.PRNGKey(0), gru_initial_state(cfg.PURE_RNN["hidden_sizes"]),
                              jnp.zeros(nx + cfg.NU + nx + cfg.NU))
        print("no trained params; using fresh init (noise scale near init may differ from late training)")

    B = cfg.PURE_RNN["batch_size"]
    key = jax.random.PRNGKey(cfg.EVAL_SEED)
    print(f"batch_size B = {B};  measuring noise scale over {BATCH} samples per horizon\n")
    print(f"  {'H':>6}  {'noise_scale':>12}  {'trainable (B_noise<=B)':>22}")
    rows = []
    for H in HORIZONS:
        if H > T:
            continue
        sample_grad = make_sample_grad(cfg, mjx_model_nominal, nominal_body_mass, network, x_refs, u_refs, H)
        batched = jax.jit(jax.vmap(sample_grad, in_axes=(None, 0, 0, 0)))
        key, ik, tk, *thk = jax.random.split(key, BATCH + 3)
        idxs = jax.random.randint(ik, (BATCH,), 0, n_traj)
        t0s = jax.random.randint(tk, (BATCH,), 0, T - H + 1)
        grads = batched(params, jnp.stack(thk), idxs, t0s)
        ns = noise_scale(grads)
        rows.append((H, ns))
        print(f"  {H:>6}  {ns:>12.1f}  {'yes' if ns <= B else 'NO':>22}")

    # linear-interpolate the crossover H where noise_scale == B (in log-noise space)
    cross = None
    for (h0, n0), (h1, n1) in zip(rows, rows[1:]):
        if n0 <= B < n1:
            f = (jnp.log(B) - jnp.log(n0)) / (jnp.log(n1) - jnp.log(n0))
            cross = float(h0 + f * (h1 - h0))
            break
    if cross is not None:
        print(f"\npredicted max trainable horizon ~ {cross:.0f} steps  (B_noise crosses batch size {B})")
    else:
        print(f"\nno crossover within {HORIZONS}: all horizons are "
              f"{'trainable' if rows[-1][1] <= B else 'noise-dominated'} at B={B}")


if __name__ == "__main__":
    main()
