"""Behavior cloning of a computed-torque expert into a GRU residual controller.

Roll the inverse-dynamics expert closed-loop on a DR-sampled plant to get an on-expert
state trajectory; at each visited state regress the GRU residual v toward
    label = expert_torque - u_ref - PD,
so that at deployment u = u_ref + PD + v reproduces the expert. The GRU sees the current
state plus an n_future-step preview of the reference (states + controls). The sim is
detached: it is a data source for states, never a gradient path, so BPTT runs through the
GRU recurrence only, not the physics.
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
LOSS_CAP = 1.0e4   # trash a whole trajectory whose mean loss exceeds this
STATE_CLIP = 1.0e3 # sanitize the per-step state so a divergent rollout can't poison v/label


def pad_reference(x_ref_full, u_ref_seq, w):
    """Right-pad the reference arrays so a length-w preview slice is valid at every t."""
    x_pad = jnp.concatenate([x_ref_full, jnp.repeat(x_ref_full[-1:], w - 1, axis=0)], axis=0)
    u_pad = jnp.concatenate([u_ref_seq, jnp.repeat(u_ref_seq[-1:], w - 1, axis=0)], axis=0)
    return x_pad, u_pad


def window_input(x_curr, x_ref_pad, u_ref_pad, t, w, nq, nu):
    """GRU per-step input: current state + w-step preview of (x_ref, u_ref)."""
    x_win = jax.lax.dynamic_slice(x_ref_pad, (t, 0), (w, 2 * nq))
    u_win = jax.lax.dynamic_slice(u_ref_pad, (t, 0), (w, nu))
    return jnp.concatenate([x_curr, x_win.reshape(-1), u_win.reshape(-1)])


def sanitize(q, qd):
    q = jnp.clip(jnp.nan_to_num(q), -STATE_CLIP, STATE_CLIP)
    qd = jnp.clip(jnp.nan_to_num(qd), -STATE_CLIP, STATE_CLIP)
    return q, qd


def make_net_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, network, h0, w, x_refs, u_refs):
    """Vmappable eval_fn(params, theta_key, idx) -> (endpoint, tracking, vrms).

    Full-T closed-loop rollout under u = u_ref + PD + v, with the same windowed input the
    GRU is trained on. params is an argument so the fn compiles once.
    """
    T = x_refs.shape[1] - 1
    nq, nu = cfg.NQ, cfg.NU
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(params, theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)
        x_ref_full = x_refs[idx]
        u_ref_seq = u_refs[idx]
        x_ref_pad, u_ref_pad = pad_reference(x_ref_full, u_ref_seq, w)
        x0 = x_ref_full[0]
        d0 = rollout.make_initial_data(model, x0[:nq], x0[nq:])

        def step(carry, t):
            d, h = carry
            q, qd = sanitize(d.qpos, d.qvel)
            x_curr = jnp.concatenate([q, qd])
            x_ref_t, u_ref_t = x_ref_full[t], u_ref_seq[t]
            new_h, v = network.apply(params, h, window_input(x_curr, x_ref_pad, u_ref_pad, t, w, nq, nu))
            pd = kp * (x_ref_t[:nq] - q) + kd * (x_ref_t[nq:] - qd)
            d = mjx.step(model, d.replace(ctrl=u_ref_t + pd + v))
            return (d, new_h), (x_curr, v)

        (final_d, _), (xs, vs) = jax.lax.scan(step, (d0, h0), jnp.arange(T))
        x_final = jnp.concatenate([final_d.qpos, final_d.qvel])
        return evaluation._rollout_metrics(xs, vs, x_final, x_refs[idx], nq)

    return eval_fn


def make_eval_callback(cfg, network, h0, w, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
                       csv_path, best_params_path):
    """Periodic closed-loop eval: log CSV, save best-endpoint params. PD baseline computed once."""
    theta_keys, idxs = evaluation._eval_plants(cfg, x_refs.shape[0], N_EVAL)

    net_eval = jax.jit(jax.vmap(
        make_net_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, network, h0, w, x_refs, u_refs),
        in_axes=(None, 0, 0)))
    pd_eval = jax.jit(jax.vmap(evaluation.make_rnn_eval_fn(
        cfg, mjx_model_nominal, nominal_body_mass,
        evaluation.pd_controller_apply(cfg.NU), jnp.zeros(1), x_refs, u_refs),
        in_axes=(None, 0, 0)))
    ep_pd = float(pd_eval(jnp.zeros(1), theta_keys, idxs)[0].mean())

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("iter,endpoint_pd,endpoint_rnn,tracking_rms_rnn,vrms_rnn,is_best\n")
    state = {"best": float("inf")}

    def callback(params, iteration):
        ep, tr, vr = net_eval(params, theta_keys, idxs)
        evaluation.summarize(ep, tr, vr, "bc_rnn")
        ep_m, tr_rms, vr_m = float(ep.mean()), float(np.sqrt(tr).mean()), float(vr.mean())
        is_best = evaluation._update_best(ep_m, state, params, best_params_path)
        with open(csv_path, "a") as f:
            f.write(f"{iteration},{ep_pd:.6f},{ep_m:.6f},{tr_rms:.6f},{vr_m:.6f},{int(is_best)}\n")
        reduction = 100 * (1 - ep_m / ep_pd) if ep_pd > 0 else float("nan")
        suffix = "  *** BEST ***" if is_best else ""
        print(f"  [eval iter {iteration:5d}] bc_rnn endpoint={ep_m:.4f} vs pd {ep_pd:.4f} "
              f"({reduction:.1f}% reduction){suffix}")

    return callback


def main():
    cfg = training.load_config()
    nq, nu = cfg.NQ, cfg.NU
    dt = cfg.TIMESTEP
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    hp = cfg.PURE_RNN
    hidden_sizes = hp["hidden_sizes"]
    w = hp["n_future"]
    batch_size = hp["batch_size"]
    lr = hp["lr"]
    n_iter = hp["n_iterations_supervised"]
    grad_clip = hp["grad_clip_norm"]

    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    N = x_refs.shape[0]
    T = u_refs.shape[1]
    print(f"  N={N} trajectories, T={T} steps")
    print(f"  hidden={hidden_sizes}, n_future={w}, batch_size={batch_size}, lr={lr}, n_iter={n_iter}")
    print(f"  behavior cloning over full T={T} (sim detached); lower batch_size if OOM")

    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    network = networks.GRUPureController(hidden_sizes=hidden_sizes, out_dim=nu)
    h0 = networks.gru_initial_state(hidden_sizes)
    step_in_dim = 2 * nq * (1 + w) + nu * w  # [x_curr, x_ref preview, u_ref preview]

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, h0, jnp.zeros(step_in_dim))

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    opt_state = optimizer.init(params)

    def per_example_loss(params, idx, theta_key):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_full = x_refs[idx]                          # (T+1, 2nq)
        u_ref_seq = u_refs[idx]                           # (T, nu)
        x_ref_seq = x_ref_full[:T]                        # (T, 2nq)
        qd_ref = x_ref_full[:, nq:]
        qddot_ref = (qd_ref[1:T + 1] - qd_ref[:T]) / dt   # (T, nq) reference accel target
        x_ref_pad, u_ref_pad = pad_reference(x_ref_full, u_ref_seq, w)

        x0 = x_ref_full[0]
        d0 = rollout.make_initial_data(model, x0[:nq], x0[nq:])

        def step(carry, inp):
            d, h = carry
            t, x_ref_t, u_ref_t, qddot_ref_t = inp
            q, qd = sanitize(d.qpos, d.qvel)
            x_curr = jnp.concatenate([q, qd])

            new_h, v = network.apply(params, h, window_input(x_curr, x_ref_pad, u_ref_pad, t, w, nq, nu))

            pd = kp * (x_ref_t[:nq] - q) + kd * (x_ref_t[nq:] - qd)
            tau = training.inverse_dynamics(model, q, qd, qddot_ref_t)
            label = jax.lax.stop_gradient(tau - u_ref_t - pd)   # residual on top of u_ref + pd
            loss_t = jnp.mean((v - label) ** 2)

            u = jax.lax.stop_gradient(tau)                      # BC: step the sim with the expert
            d = mjx.step(model, d.replace(ctrl=u))
            return (d, new_h), loss_t

        _, losses_t = jax.lax.scan(
            step, (d0, h0), (jnp.arange(T), x_ref_seq, u_ref_seq, qddot_ref))
        return jnp.mean(losses_t[RNN_WARMUP:])

    batched_loss = jax.vmap(per_example_loss, in_axes=(None, 0, 0))

    def loss_fn(params, idxs, theta_keys):
        loss_arr = batched_loss(params, idxs, theta_keys)
        valid = jax.lax.stop_gradient(loss_arr < LOSS_CAP)      # trash whole exploded trajectories
        loss = jnp.sum(jnp.where(valid, loss_arr, 0.0)) / jnp.maximum(jnp.sum(valid), 1.0)
        return loss, 1.0 - jnp.mean(valid.astype(jnp.float32))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(params, opt_state, idxs, theta_keys):
        (loss, frac_masked), grads = grad_fn(params, idxs, theta_keys)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, frac_masked

    print("building closed-loop eval ...")
    params_path = cfg.OUTPUT_DIR / "pure_rnn_params.pkl"
    eval_callback = make_eval_callback(
        cfg, network, h0, w, x_refs, u_refs, mjx_model_nominal, nominal_body_mass,
        csv_path=cfg.OUTPUT_DIR / "pure_rnn_eval_log.csv",
        best_params_path=params_path)

    print(f"training {n_iter} iterations ...")
    loss_history = np.zeros(n_iter)
    for i in range(n_iter):
        key, k_idx, k_theta = jax.random.split(key, 3)
        idxs = jax.random.randint(k_idx, (batch_size,), 0, N)
        theta_keys = jax.random.split(k_theta, batch_size)

        params, opt_state, loss, frac_masked = train_step(params, opt_state, idxs, theta_keys)
        loss_history[i] = float(loss)

        if (i + 1) % LOG_EVERY == 0:
            print(f"  iter {i + 1:5d}/{n_iter}  loss={loss:.4f}  masked={float(frac_masked):.2%}")
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
