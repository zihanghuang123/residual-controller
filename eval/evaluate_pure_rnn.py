"""Full hold-out eval, PD vs pure_rnn (no two-model).

Mirrors evaluate_pure.py but the residual controller is the GRU, so this uses
rollout_rnn (no history window). PD shares the same rollout with a 1-element
dummy h threaded through unchanged, so its numbers match evaluate_pure.py's pd.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from lib import evaluation, losses, rollout, training
from lib.domain_randomization import apply_theta, sample_theta
from lib.networks import GRUPureController, gru_initial_state


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, controller_fn, h0, x_refs, u_refs):
    """Vmappable eval_fn(theta_key, idx) -> (endpoint, tracking_mse, vrms)."""
    T = x_refs.shape[1] - 1
    nq = cfg.NQ
    kp = jnp.asarray(cfg.KP)
    kd = jnp.asarray(cfg.KD)

    def eval_fn(theta_key, idx):
        theta = sample_theta(theta_key, cfg.N_LINKS, cfg.DR_RANGES)
        mjx_model = apply_theta(mjx_model_nominal, theta, nominal_body_mass, cfg.N_LINKS)

        x_ref_full = x_refs[idx, :T]
        u_ref_full = u_refs[idx]
        x_target = x_refs[idx, -1]

        xs, _us, vs, x_final = rollout.rollout_rnn(
            mjx_model, x_ref_full[0], x_ref_full, u_ref_full,
            h0, controller_fn, kp, kd, T)
        xs_full = jnp.concatenate([xs, x_final[None]], axis=0)

        endpoint = losses.endpoint_error(x_final, x_target, nq)
        tracking = losses.tracking_loss(xs_full, x_refs[idx], nq)
        vrms = jnp.sqrt(jnp.mean(jnp.sum(vs ** 2, axis=-1)))
        return endpoint, tracking, vrms

    return eval_fn


def make_pd_controller(cfg):
    def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
        return h, jnp.zeros(cfg.NU)
    return controller_fn


def make_pure_rnn_controller(network, params):
    def controller_fn(h, x_curr, u_prev, x_ref, u_ref):
        return network.apply(params, h, rollout.make_rnn_step_input(x_curr, u_prev, x_ref, u_ref))
    return controller_fn


def main():
    cfg = training.load_config()

    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    n_traj = x_refs.shape[0]
    T = x_refs.shape[1] - 1
    print(f"  {n_traj} converged trajectories of length T={T}  (hidden={cfg.PURE_RNN['hidden_sizes']})")

    print("loading pure_rnn params + building network ...")
    pure_params = load_pkl(cfg.OUTPUT_DIR / "pure_rnn_params.pkl")
    pure_network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    print(f"sampling {cfg.N_EVAL_PLANTS} eval plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    theta_keys = jax.random.split(jax.random.PRNGKey(cfg.EVAL_SEED), cfg.N_EVAL_PLANTS)
    idxs = jnp.arange(cfg.N_EVAL_PLANTS) % n_traj

    controllers = {
        "pd": (make_pd_controller(cfg), jnp.zeros(1)),
        "pure_rnn": (make_pure_rnn_controller(pure_network, pure_params),
                     gru_initial_state(cfg.PURE_RNN["hidden_sizes"])),
    }

    results = {}
    print("evaluating ...")
    for name, (controller_fn, h0) in controllers.items():
        eval_fn = make_eval_fn(cfg, mjx_model_nominal, nominal_body_mass, controller_fn, h0, x_refs, u_refs)
        endpoints, trackings, vrmss = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))(theta_keys, idxs)
        results[name] = (np.asarray(endpoints), np.asarray(trackings), np.asarray(vrmss))
        evaluation.summarize(endpoints, trackings, vrmss, name)

    ep_pd = results["pd"][0].mean()
    ep_pure_rnn = results["pure_rnn"][0].mean()
    print(f"\npure_rnn reduces mean endpoint by {100 * (1 - ep_pure_rnn / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_pure_rnn:.3f})")

    evaluation.save_metrics(results, cfg.OUTPUT_DIR / "metrics_pure_rnn.npz")


if __name__ == "__main__":
    main()
