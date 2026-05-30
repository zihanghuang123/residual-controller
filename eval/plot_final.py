"""Generate training-curve and eval-metric figures from cfg.OUTPUT_DIR.

Reads whichever loss histories and metric files exist and writes:
    training_curves.png  — log-y loss vs iteration for each available training run
    eval_metrics.png     — endpoint + tracking error distributions (box + histogram)

Skips controllers whose loss/metric files aren't present.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from lib import training

# Loss histories to look for. {label: filename}.
LOSS_FILES = {
    "PURE controller": "pure_loss_history.npy",
    "ORACLE controller": "oracle_loss_history.npy",
    "theta estimator": "theta_loss_history.npy",
    "two-model controller": "controller_loss_history.npy",
}

# Controllers that appear in metrics.npz (or metrics_pure.npz / metrics_oracle.npz).
CONTROLLER_COLORS = {
    "pd": "tab:gray",
    "pure": "tab:orange",
    "oracle": "tab:green",
    "two_model": "tab:blue",
}


def plot_loss_curve(loss_history, ax, title):
    ax.plot(loss_history)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def plot_distribution(values_dict, ax, title, ylabel):
    names = list(values_dict.keys())
    data = [values_dict[n] for n in names]
    bp = ax.boxplot(data, labels=names, showfliers=True, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(CONTROLLER_COLORS.get(name, "tab:gray"))
        patch.set_alpha(0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_histogram(values_dict, ax, title, xlabel, bins=40):
    all_vals = np.concatenate(list(values_dict.values()))
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    for name, vals in values_dict.items():
        ax.hist(vals, bins=bin_edges, alpha=0.5, label=name,
                color=CONTROLLER_COLORS.get(name, "tab:gray"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def make_training_figure(output_dir, save_path):
    """One panel per available loss history."""
    available = {label: output_dir / fname for label, fname in LOSS_FILES.items()
                 if (output_dir / fname).exists()}
    if not available:
        print("no loss history files found — skipping training_curves.png")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (label, path) in zip(axes[0], available.items()):
        plot_loss_curve(np.load(path), ax, label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}  ({n} panel(s): {list(available.keys())})")


def collect_metrics(output_dir):
    """Merge metrics from metrics.npz / metrics_pure.npz / metrics_oracle.npz if any exist.

    Returns {controller_name: {'endpoint': arr, 'tracking': arr}}.
    """
    by_controller = {}
    for fname in ("metrics.npz", "metrics_pure.npz", "metrics_oracle.npz"):
        path = output_dir / fname
        if not path.exists():
            continue
        data = np.load(path)
        for key in data.files:
            # Keys are endpoint_<name>, tracking_<name>, vrms_<name>
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            metric, name = parts
            if metric not in ("endpoint", "tracking"):
                continue
            by_controller.setdefault(name, {})[metric] = data[key]
    return by_controller


def make_eval_figure(output_dir, save_path):
    """2x2 grid: endpoint (box, hist) on top, tracking (box, hist) on bottom."""
    by_controller = collect_metrics(output_dir)
    if not by_controller:
        print("no metrics npz files found — skipping eval_metrics.png")
        return

    endpoint = {n: by_controller[n]["endpoint"]
                for n in by_controller if "endpoint" in by_controller[n]}
    tracking_rms = {n: np.sqrt(by_controller[n]["tracking"])
                    for n in by_controller if "tracking" in by_controller[n]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_distribution(endpoint, axes[0, 0], "endpoint error (box)", "L2 distance")
    plot_histogram(endpoint, axes[0, 1], "endpoint error (hist)", "L2 distance")
    plot_distribution(tracking_rms, axes[1, 0], "tracking error RMS (box)", "RMS distance")
    plot_histogram(tracking_rms, axes[1, 1], "tracking error RMS (hist)", "RMS distance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}  (controllers: {list(endpoint.keys())})")


def main():
    cfg = training.load_config()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_training_figure(cfg.OUTPUT_DIR, cfg.OUTPUT_DIR / "training_curves.png")
    make_eval_figure(cfg.OUTPUT_DIR, cfg.OUTPUT_DIR / "eval_metrics.png")


if __name__ == "__main__":
    main()
