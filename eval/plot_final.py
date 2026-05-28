"""Generate training-curve and eval-metric figures from outputs/double_pendulum/.

Reads loss histories and metrics.npz, writes two PNGs into the same directory:
    training_curves.png  — log-y loss vs iteration for PURE, THETA, CONTROLLER
    eval_metrics.png     — endpoint + tracking error distributions (box + histogram)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "double_pendulum"
PURE_LOSS_PATH = OUTPUT_DIR / "pure_loss_history.npy"
THETA_LOSS_PATH = OUTPUT_DIR / "theta_loss_history.npy"
CONTROLLER_LOSS_PATH = OUTPUT_DIR / "controller_loss_history.npy"
METRICS_PATH = OUTPUT_DIR / "metrics.npz"

TRAINING_FIG_PATH = OUTPUT_DIR / "training_curves.png"
EVAL_FIG_PATH = OUTPUT_DIR / "eval_metrics.png"

CONTROLLER_NAMES = ["pd", "pure", "two_model"]
CONTROLLER_COLORS = {"pd": "tab:gray", "pure": "tab:orange", "two_model": "tab:blue"}


def plot_loss_curve(loss_history: np.ndarray, ax: plt.Axes, title: str) -> None:
    """Single log-y loss curve on the given axes."""
    ax.plot(loss_history)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def plot_distribution(values_dict: dict, ax: plt.Axes, title: str, ylabel: str) -> None:
    """Side-by-side box plot for {controller_name: 1D array}."""
    names = list(values_dict.keys())
    data = [values_dict[n] for n in names]
    bp = ax.boxplot(data, labels=names, showfliers=True, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(CONTROLLER_COLORS.get(name, "tab:gray"))
        patch.set_alpha(0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_histogram(values_dict: dict, ax: plt.Axes, title: str, xlabel: str, bins: int = 40) -> None:
    """Overlapped histograms for {controller_name: 1D array}."""
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


def make_training_figure(save_path: Path) -> None:
    """Three-panel figure: PURE, THETA, CONTROLLER loss curves."""
    pure_loss = np.load(PURE_LOSS_PATH)
    theta_loss = np.load(THETA_LOSS_PATH)
    controller_loss = np.load(CONTROLLER_LOSS_PATH)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_loss_curve(pure_loss, axes[0], "PURE controller")
    plot_loss_curve(theta_loss, axes[1], "theta estimator")
    plot_loss_curve(controller_loss, axes[2], "two-model controller")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def make_eval_figure(save_path: Path) -> None:
    """2x2 grid: endpoint (box, hist) on top, tracking (box, hist) on bottom."""
    data = np.load(METRICS_PATH)
    endpoint = {n: data[f"endpoint_{n}"] for n in CONTROLLER_NAMES}
    tracking_rms = {n: np.sqrt(data[f"tracking_{n}"]) for n in CONTROLLER_NAMES}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_distribution(endpoint, axes[0, 0], "endpoint error (box)", "L2 distance")
    plot_histogram(endpoint, axes[0, 1], "endpoint error (hist)", "L2 distance")
    plot_distribution(tracking_rms, axes[1, 0], "tracking error RMS (box)", "RMS distance")
    plot_histogram(tracking_rms, axes[1, 1], "tracking error RMS (hist)", "RMS distance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"saved {save_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_training_figure(TRAINING_FIG_PATH)
    make_eval_figure(EVAL_FIG_PATH)


if __name__ == "__main__":
    main()
