"""On-policy DAgger for the GRU residual controller with per-step online updates.

Rolls the batch closed-loop (sim detached) and, at each timestep, updates the GRU on the
batch-mean per-step loss toward the inverse-dynamics expert label. Truncated BPTT length 1
(hidden state detached across steps). State is sanitized + exploded examples masked per step.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mujoco import mjx

from lib import evaluation, networks, rollout, training
from lib.domain_randomization import apply_theta, sample_theta

RNN_WARMUP = 50    # skip the cold-start transient when reporting the loss
LOG_EVERY = 1
EVAL_EVERY = 10    # closed-loop eval cadence, in rollouts; None disables
N_EVAL = 200
BETA_DECAY_ITERS = 10000   # DAgger: roll expert, anneal beta 1->0 onto the learner over this many rollouts
LOSS_CAP = 1.0e4   # mask a step's example whose per-step loss exceeds this (explosion)
STATE_CLIP = 1.0e3 # sanitize the per-step state so a divergent rollout can't poison v/label with inf/nan


def main():
    cfg = training.load_config()
    nq, nu = cfg.NQ, cfg.NU
    dt = cfg.TIMESTEP

    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    N = x_refs.shape[0]
    T = u_refs.shape[1]
    print(f"  N={N} trajectories, T={T} steps")

    hp = cfg.PURE_RNN
    hidden_sizes = hp["hidden_sizes"]
    batch_size = hp["batch_size"]
    lr = hp["lr"]
    n_iter = hp["n_iterations"]
    grad_clip = hp["grad_clip_norm"]
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    print(f"  hidden={hidden_sizes}, batch_size={batch_size}, lr={lr}, n_iter={n_iter}")
    print(f"  per-step online updates over T={T} (sim detached, TBPTT len 1)")

    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    network = networks.GRUPureController(hidden_sizes=hidden_sizes, out_dim=nu)
    h0 = networks.gru_initial_state(hidden_sizes)
    dummy_x = rollout.make_rnn_step_input(jnp.zeros(2 * nq), jnp.zeros(nu), jnp.zeros(2 * nq), jnp.zeros(nu))

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, h0, dummy_x)

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optimizer.init(params)

    def inverse_feedforward(model, q, qd, qddot):
        d = mjx.make_data(model)
        d = d.replace(qpos=q, qvel=qd, qacc=qddot)
        return mjx.inverse(model, d).qfrc_inverse

    @jax.jit
    def train_rollout(params, opt_state, x_refs, u_refs, idxs, theta_keys, beta):
        thetas = jax.vmap(lambda k: sample_theta(k, cfg.N_LINKS, cfg.DR_RANGES))(theta_keys)
        models = jax.vmap(apply_theta, in_axes=(None, 0, None, None))(
            mjx_model_nominal, thetas, nominal_body_mass, cfg.N_LINKS)

        x_seq = x_refs[idxs, :T]                                  # (B, T, 2nq)
        u_seq = u_refs[idxs]                                      # (B, T, nu)
        qddot_seq = (x_refs[idxs, 1:T + 1, nq:] - x_refs[idxs, :T, nq:]) / dt
        x_t = jnp.transpose(x_seq, (1, 0, 2))                     # scan over time -> (T, B, ...)
        u_t = jnp.transpose(u_seq, (1, 0, 2))
        qa_t = jnp.transpose(qddot_seq, (1, 0, 2))

        bs = idxs.shape[0]
        d0 = jax.vmap(rollout.make_initial_data)(models, x_seq[:, 0, :nq], x_seq[:, 0, nq:])
        h0_b = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (bs,) + a.shape), h0)

        def step(carry, inp):
            d, h, params, opt_state, u_prev = carry
            x_ref, u_ref, qddot_ref = inp

            q = jnp.clip(jnp.nan_to_num(d.qpos), -STATE_CLIP, STATE_CLIP)
            qd = jnp.clip(jnp.nan_to_num(d.qvel), -STATE_CLIP, STATE_CLIP)
            x_curr = jnp.concatenate([q, qd], axis=-1)
            x_in = jax.vmap(rollout.make_rnn_step_input)(x_curr, u_prev, x_ref, u_ref)

            pd = kp * (x_ref[:, :nq] - q) + kd * (x_ref[:, nq:] - qd)
            tau = jax.vmap(inverse_feedforward)(models, q, qd, qddot_ref)
            label = jax.lax.stop_gradient(tau - u_ref - pd)

            def batch_loss(p):
                new_h, v = jax.vmap(lambda hh, xx: network.apply(p, hh, xx))(
                    jax.lax.stop_gradient(h), x_in)
                se = jnp.mean((v - label) ** 2, axis=-1)         # (B,) per-example
                valid = jax.lax.stop_gradient(se < LOSS_CAP)
                loss = jnp.sum(jnp.where(valid, se, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
                return loss, (new_h, v, valid)

            (loss, (new_h, v, valid)), grads = jax.value_and_grad(batch_loss, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            residual = beta * label + (1.0 - beta) * v
            u = jax.lax.stop_gradient(u_ref + pd + residual)
            d = jax.vmap(lambda m, dd, uu: mjx.step(m, dd.replace(ctrl=uu)))(models, d, u)
            frac = 1.0 - jnp.mean(valid.astype(jnp.float32))
            return (d, new_h, params, opt_state, u), (loss, frac)

        init = (d0, h0_b, params, opt_state, jnp.zeros((bs, nu)))
        (_, _, params, opt_state, _), (losses, fracs) = jax.lax.scan(step, init, (x_t, u_t, qa_t))
        return params, opt_state, jnp.mean(losses[RNN_WARMUP:]), jnp.mean(fracs[RNN_WARMUP:])

    print("building closed-loop eval ...")
    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    eval_callback = evaluation.make_rnn_eval_callback(
        cfg, network, h0, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
        csv_path=cfg.OUTPUT_DIR / "pure_rnn_eval_log.csv",
        best_params_path=params_path, n_eval=N_EVAL)

    print(f"training {n_iter} rollouts ({T} updates each) ...")
    loss_history = np.zeros(n_iter)
    for i in range(n_iter):
        key, k_idx, k_theta = jax.random.split(key, 3)
        idxs = jax.random.randint(k_idx, (batch_size,), 0, N)
        theta_keys = jax.random.split(k_theta, batch_size)

        beta = max(0.0, 1.0 - i / BETA_DECAY_ITERS)
        params, opt_state, loss, frac = train_rollout(
            params, opt_state, x_refs, u_refs, idxs, theta_keys, jnp.float32(beta))
        loss_history[i] = float(loss)

        if (i + 1) % LOG_EVERY == 0:
            print(f"  rollout {i + 1:4d}/{n_iter}  loss={float(loss):.4f}  "
                  f"masked={float(frac):.2%}  beta={beta:.3f}")
        if EVAL_EVERY and (i + 1) % EVAL_EVERY == 0:
            eval_callback(params, i + 1)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = cfg.OUTPUT_DIR / "pure_rnn_loss_history.npy"
    if not params_path.exists():
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
    np.save(loss_path, loss_history)
    print(f"saved {loss_path}; best closed-loop params at {params_path}")


if __name__ == "__main__":
    main()
