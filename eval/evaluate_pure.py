"""Full hold-out eval, PD vs pure only (skips the two-model stack).

Same N_EVAL_PLANTS / EVAL_SEED protocol as evaluate.py, but restricted to the two controllers that don't need the theta estimator.
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax.numpy as jnp

from lib import evaluation, rollout, training
from lib.networks import MLPPureController


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_pd_factory(cfg):
    def make_controller_fn(_theta):
        return lambda *_: jnp.zeros(cfg.NU)
    return make_controller_fn


def make_pure_factory(network, params):
    def make_controller_fn(_theta):
        def controller_fn(x_hist_full, u_hist, x_ref_window, u_ref_window):
            net_in = rollout.make_network_input(x_hist_full, u_hist, x_ref_window, u_ref_window)
            return network.apply(params, net_in)
        return controller_fn
    return make_controller_fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plants/double_pendulum/config.py")
    parser.add_argument("--no-pd", action="store_true",
                        help="Disable PD baseline (use for supervised-trained controllers).")
    args, _ = parser.parse_known_args()
    import sys as _sys
    _sys.argv = [_sys.argv[0], "--config", args.config]
    cfg = training.load_config()

    print("loading pure params + building network ...")
    pure_params = load_pkl(cfg.OUTPUT_DIR / "pure_params.pkl")
    pure_network = MLPPureController(hidden_sizes=cfg.PURE["hidden_sizes"], out_dim=cfg.NU)

    controllers = {
        "pd": make_pd_factory(cfg),
        "pure": make_pure_factory(pure_network, pure_params),
    }

    results = evaluation.eval_mlp_residual(
        cfg, controllers,
        traj_path=cfg.OUTPUT_DIR / "trajectories.npz",
        metrics_path=cfg.OUTPUT_DIR / "metrics_pure.npz",
        w=cfg.PURE["n_history"],
        use_pd=not args.no_pd,
    )

    ep_pd = results["pd"][0].mean()
    ep_pure = results["pure"][0].mean()
    print(f"\npure reduces mean endpoint by {100 * (1 - ep_pure / ep_pd):.1f}% vs pd "
          f"({ep_pd:.3f} -> {ep_pure:.3f})")


if __name__ == "__main__":
    main()
