"""Hold-out eval, PD vs pure_rnn, via rollout_rnn (no history window)."""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from lib import evaluation, training
from lib.networks import GRUPureController, gru_initial_state


def main():
    cfg = training.load_config()

    print("loading trajectories ...")
    x_refs, u_refs = training.load_trajectories(cfg.OUTPUT_DIR / "trajectories.npz")
    T = x_refs.shape[1] - 1
    print(f"  {x_refs.shape[0]} converged trajectories of length T={T}  (hidden={cfg.PURE_RNN['hidden_sizes']})")

    print("loading pure_rnn params + building network ...")
    with open(cfg.OUTPUT_DIR / "pure_rnn_params.pkl", "rb") as f:
        pure_params = pickle.load(f)
    network = GRUPureController(hidden_sizes=cfg.PURE_RNN["hidden_sizes"], out_dim=cfg.NU)
    mjx_model_nominal, nominal_body_mass = training.build_mjx_model(cfg.MODEL_PATH)

    controllers = {
        "pd": (evaluation.pd_controller_apply(cfg.NU), jnp.zeros(1), jnp.zeros(1)),
        "pure_rnn": (evaluation.rnn_controller_apply(network), pure_params,
                     gru_initial_state(cfg.PURE_RNN["hidden_sizes"])),
    }

    print(f"evaluating {cfg.N_EVAL_PLANTS} plants under EVAL_SEED={cfg.EVAL_SEED} ...")
    results = evaluation.evaluate_rnn_controllers(
        cfg, controllers, x_refs, u_refs, mjx_model_nominal, nominal_body_mass)

    ep_pd = results["pd"][0].mean()
    ep_rnn = results["pure_rnn"][0].mean()
    print(f"\npure_rnn reduces mean endpoint by {100 * (1 - ep_rnn / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_rnn:.3f})")

    evaluation.save_metrics(results, cfg.OUTPUT_DIR / "metrics_pure_rnn.npz")


if __name__ == "__main__":
    main()
