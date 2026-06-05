"""On-policy DAgger for the GRU residual controller.

Roll the controller closed-loop with the simulator detached (stop_gradient on the action),
label each visited state with the inverse-dynamics feedforward inverse(theta, q, qd, qddot_ref) - u_ref,
and regress the GRU toward it. Fixes covariate shift without backprop through the chaotic sim.
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

RNN_WARMUP = 50    # skip the cold-start transient (zero-init h) when scoring the loss
LOG_EVERY = 50
EVAL_EVERY = 500   # closed-loop eval cadence; None disables
N_EVAL = 200
BETA_DECAY_ITERS = 6000   # DAgger: roll out expert, anneal beta 1->0 onto the learner over this many iters
LOSS_CAP = 1.0e4     # trash a whole trajectory whose mean loss exceeds this (healthy is < ~hundreds)
STATE_CLIP = 1.0e3   # sanitize the per-step state so a divergent rollout can't poison v/label with inf/nan


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
    n_iter = hp["n_iterations_supervised"]
    grad_clip = hp["grad_clip_norm"]
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)
    print(f"  hidden={hidden_sizes}, batch_size={batch_size}, lr={lr}, n_iter={n_iter}")
    print(f"  on-policy rollout over T={T} (sim detached); lower batch_size if OOM")

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

    def per_example_loss(params, x_refs, u_refs, idx, theta_key, beta):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_seq = x_refs[idx, :T]                       # (T, 2nq)
        u_ref_seq = u_refs[idx]                           # (T, nu)
        qd_ref = x_refs[idx, :, nq:]
        qddot_ref_seq = (qd_ref[1:] - qd_ref[:T]) / dt    # (T, nq) reference accel target

        x0 = x_refs[idx, 0]
        d0 = rollout.make_initial_data(model, x0[:nq], x0[nq:])

        def step(carry, inp):
            d, h, u_prev = carry
            x_ref, u_ref, qddot_ref = inp
            q_curr = jnp.clip(jnp.nan_to_num(d.qpos), -STATE_CLIP, STATE_CLIP)
            qd_curr = jnp.clip(jnp.nan_to_num(d.qvel), -STATE_CLIP, STATE_CLIP)
            x_curr = jnp.concatenate([q_curr, qd_curr])

            step_in = rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref)
            new_h, v = network.apply(params, h, step_in)

            pd = kp * (x_ref[:nq] - q_curr) + kd * (x_ref[nq:] - qd_curr)
            tau = inverse_feedforward(model, q_curr, qd_curr, qddot_ref)
            label = jax.lax.stop_gradient(tau - u_ref - pd)      # residual on top of u_ref + pd
            loss_t = jnp.mean((v - label) ** 2)

            residual = beta * label + (1.0 - beta) * v          # DAgger: roll expert, anneal to learner
            u = jax.lax.stop_gradient(u_ref + pd + residual)    # detach: sim is a data source, not a grad path
            d = mjx.step(model, d.replace(ctrl=u))
            return (d, new_h, u), loss_t

        _, losses_t = jax.lax.scan(
            step, (d0, h0, jnp.zeros(nu)), (x_ref_seq, u_ref_seq, qddot_ref_seq))
        return jnp.mean(losses_t[RNN_WARMUP:])

    # Dataset passed as args (None axis), not closed over, so XLA doesn't bake it in.
    batched_loss = jax.vmap(per_example_loss, in_axes=(None, None, None, 0, 0, None))

    def loss_fn(params, x_refs, u_refs, idxs, theta_keys, beta):
        loss_arr = batched_loss(params, x_refs, u_refs, idxs, theta_keys, beta)
        valid = jax.lax.stop_gradient(loss_arr < LOSS_CAP)       # trash whole exploded trajectories
        loss = jnp.sum(jnp.where(valid, loss_arr, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
        return loss, 1.0 - jnp.mean(valid.astype(jnp.float32))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, opt_state, x_refs, u_refs, idxs, theta_keys, beta):
        (loss, frac_masked), grads = grad_fn(params, x_refs, u_refs, idxs, theta_keys, beta)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, frac_masked

    print("building closed-loop eval ...")
    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    eval_callback = evaluation.make_rnn_eval_callback(
        cfg, network, h0, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
        csv_path=cfg.OUTPUT_DIR / "pure_rnn_eval_log.csv",
        best_params_path=params_path,
        n_eval=N_EVAL)

    print(f"training {n_iter} iterations ...")
    loss_history = np.zeros(n_iter)
    for i in range(n_iter):
        key, k_idx, k_theta = jax.random.split(key, 3)
        idxs = jax.random.randint(k_idx, (batch_size,), 0, N)
        theta_keys = jax.random.split(k_theta, batch_size)

        beta = max(0.0, 1.0 - i / BETA_DECAY_ITERS)
        params, opt_state, loss, frac_masked = train_step(
            params, opt_state, x_refs, u_refs, idxs, theta_keys, jnp.float32(beta))
        loss_history[i] = float(loss)

        if (i + 1) % LOG_EVERY == 0:
            print(f"  iter {i + 1:5d}/{n_iter}  loss={loss:.4f}  masked={float(frac_masked):.2%}  beta={beta:.3f}")
        if EVAL_EVERY and (i + 1) % EVAL_EVERY == 0:
            eval_callback(params, i + 1)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loss_path = cfg.OUTPUT_DIR / "pure_rnn_loss_history.npy"
    if not params_path.exists():   # no eval checkpoint saved; fall back to the final iterate
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
    np.save(loss_path, loss_history)
    print(f"saved {loss_path}; best closed-loop params at {params_path}")


if __name__ == "__main__":
    main()
